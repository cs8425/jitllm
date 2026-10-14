package model

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestServingSplitAB is one session decoding while another prefills, on two
// pool layouts, in one process, interleaved (RULE 2), an A/A first and the
// comparison twice:
//
//   - base: the default pools. Both sessions decode on the P-cores and the
//     prefiller prefills on the P and E cores; the sets overlap, so every
//     region of either waits its turn on one lock, and the decoder, never
//     alone on its crew, never fuses.
//   - split: the decoder on the P-cores with no prefill pool
//     (WithPrefillCoreSet("p")); the prefiller wholly on the E-cores
//     (WithCoreSet("e") and WithPrefillCoreSet("e")), its workers and its
//     driving goroutine at background QoS (sched.EQoS). Disjoint sets take
//     turns on two locks, so neither waits on the other and the decoder fuses.
//
// Each turn opens a State per session, prefills the decoder to the depth,
// starts the prefiller on plen-token prompts in a loop, and times run decoded
// tokens while it runs; the prefiller's rate is what it finished in that time.
// Decode ratios are base/split time: above 1, split decodes faster.
//
//	JITLLM_SPLIT_AB=1 JITLLM_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf [JITLLM_AB_ROUNDS=20]
func TestServingSplitAB(t *testing.T) {
	if os.Getenv("JITLLM_SPLIT_AB") == "" {
		t.Skip("set JITLLM_SPLIT_AB=1")
	}
	p := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if p == "" {
		t.Skip("set JITLLM_MODEL")
	}
	p = jlmOf(t, p)
	rounds := 20
	if n, err := strconv.Atoi(os.Getenv("JITLLM_AB_ROUNDS")); err == nil && n > 0 {
		rounds = n
	}
	open := func(o ...nn.Option) *Model {
		m, err := Open(p, noTune, WithJITOptions(o...))
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	dBase, sBase := open(), open()
	dSplit := open(nn.WithPrefillCoreSet("p"))
	sSplit := open(nn.WithSched(sched.WithCoreSet("e")), nn.WithPrefillCoreSet("e"))
	defer func() {
		for _, m := range []*Model{dBase, sBase, dSplit, sSplit} {
			m.Close()
		}
	}()
	const depth, run, plen = 64, 64, 32
	prompt := make([]int32, depth)
	for i := range prompt {
		prompt[i] = int32(1 + (i*7919)%1000)
	}
	type result struct {
		dec   time.Duration
		pre   float64 // the prefiller's tokens a second over the decode
		fused int64
	}
	type arm struct {
		name   string
		dm, sm *Model
		eqos   bool
	}
	turn := func(a arm) result {
		d := a.dm.NewState(depth + run + 64)
		defer d.Close()
		s := a.sm.NewState(plen + 8)
		defer s.Close()
		if _, err := d.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		var stop atomic.Bool
		var toks atomic.Int64
		var wg sync.WaitGroup
		started := make(chan struct{})
		wg.Go(func() {
			// The goroutine exits locked to its thread, which the runtime
			// then ends rather than reusing at background QoS.
			if a.eqos {
				sched.EQoS()
			}
			close(started)
			for !stop.Load() {
				s.Reset()
				if _, err := s.Prefill(prompt[:plen]); err != nil {
					t.Error(err)
					return
				}
				toks.Add(plen)
			}
		})
		<-started
		time.Sleep(20 * time.Millisecond)
		n0, r0 := toks.Load(), d.FusedRuns
		t0 := time.Now()
		for i := range run {
			if _, err := d.ForwardGreedy(int32(1 + i%100)); err != nil {
				t.Fatal(err)
			}
		}
		el := time.Since(t0)
		n1 := toks.Load()
		stop.Store(true)
		wg.Wait()
		return result{dec: el, pre: float64(n1-n0) / el.Seconds(), fused: d.FusedRuns - r0}
	}
	median := func(v []float64) (float64, float64) {
		sort.Float64s(v)
		return v[len(v)/2], v[len(v)*3/4] - v[len(v)/4]
	}
	ab := func(a, b arm) string {
		turn(a)
		turn(b)
		var ratio, da, db, pa, pb []float64
		var fa, fb int64
		for r := range rounds {
			var x, y result
			if r%2 == 0 {
				x, y = turn(a), turn(b)
			} else {
				y, x = turn(b), turn(a)
			}
			ratio = append(ratio, float64(x.dec)/float64(y.dec))
			da, db = append(da, run/x.dec.Seconds()), append(db, run/y.dec.Seconds())
			pa, pb = append(pa, x.pre), append(pb, y.pre)
			fa, fb = fa+x.fused, fb+y.fused
		}
		m, iqr := median(ratio)
		mda, _ := median(da)
		mdb, _ := median(db)
		mpa, _ := median(pa)
		mpb, _ := median(pb)
		v := ""
		if iqr/m > 0.10 {
			v = "  UNSTABLE -- do not quote this"
		}
		return fmt.Sprintf("decode %s/%s time %.4fx (IQR/median %.1f%%, n=%d): %s %.1f tok/s, %s %.1f tok/s; "+
			"prefill beside it %s %.0f tok/s, %s %.0f tok/s; fused runs %d, %d%s",
			a.name, b.name, m, 100*iqr/m, rounds, a.name, mda, b.name, mdb, a.name, mpa, b.name, mpb, fa, fb, v)
	}
	base, split := arm{"base", dBase, sBase, false}, arm{"split", dSplit, sSplit, true}
	t.Logf("A/A: %s", ab(base, arm{"base-B", dBase, sBase, false}))
	for pass := 1; pass <= 2; pass++ {
		r := ab(base, split)
		t.Logf("pass %d: %s", pass, r)
	}
}
