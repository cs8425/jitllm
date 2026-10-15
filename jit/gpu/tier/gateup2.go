package tier

import (
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// gateUp2On says block l's gated FFN runs gate and up as one Gate2 launch
// (Config.GateUp2): up's kernel carries the gate's weight as a second stream
// and writes act(gate)*up where up's gated launch wrote it. It holds where
// the two launches it replaces would run -- a one-token decode, a plain dense
// gated FFN whose up has a gated form -- where gate and up are one shape at
// one split, and where this device chose it for the shape (gateUp2Pick).
func (g *devTier) gateUp2On(l *layer, R int, ragged, voltaOnly, ungated, noActMul bool) bool {
	mg, mu := &l.mvg, &l.mvu
	return g.GateUp2 && g.gateUp2Shape(l) && R == 1 && !ragged && !voltaOnly && !ungated && !noActMul &&
		(g.GateUp2Force || g.gateUp2Pick(splitKey{mu.q, mu.rows, mu.k})) &&
		!(g.TableSplit && (mg.alt != nil || mu.alt != nil))
}

// gateUp2Shape says block l's FFN has a Gate2 form: a plain dense gated FFN
// whose gate and up are one shape at one final-row split.
func (g *devTier) gateUp2Shape(l *layer) bool {
	mg, mu := &l.mvg, &l.mvu
	return !g.Persist && mu.gate2 != nil && mu.gated != nil && l.clampB == nil && l.nSubFFN == nil && !l.sparse &&
		l.router == nil && l.gate != nil && l.up != nil && mg.kern != nil && mg.bias == nil && mu.bias == nil &&
		mg.q == mu.q && mg.rows == mu.rows && mg.k == mu.k && mg.split == mu.split && mg.group == mu.group &&
		mg.rowt <= 1 && mg.slots == 0 && mg.experts <= 1
}

// gateUp2Pick is this device's choice for a shape: the tuner's verdict
// (tuneGateUp2) where it has one; otherwise Gate2 where it writes the two
// launches' bits (kernels.Gate2Exact), which measured faster on every shape
// a V100 was given, and the two launches elsewhere until a trial says.
func (g *devTier) gateUp2Pick(key splitKey) bool {
	if v, ok := g.gu2Win[key]; ok {
		return v
	}
	return kernels.Gate2Exact(g.dev.API())
}

// gateUp2Margin is how much faster than the two launches Gate2 must time to
// be taken: the trial resolves about 1%.
const gateUp2Margin = 0.99

// tuneGateUp2 times, per up-projection shape with a Gate2 form, the gate's
// launch and up's gated launch against the one Gate2 launch over a sequence
// of the placed blocks that share the shape, the way a token runs them, and
// records the winner in gu2Win. Gate2 is not bit-identical to the two
// launches off PTX (kernels.Gate2Exact) but sits in the reduction-order band,
// so on every backend the choice is speed. Cached per binary, device and
// shape, as the split tuner's verdicts are. Called from retuneDecode,
// holding g.mu, outside any Session.
func (g *devTier) tuneGateUp2() {
	if g.bs == nil || !g.GateUp2 || g.GateUp2Force {
		return
	}
	if g.gu2Win == nil {
		g.gu2Win = map[splitKey]bool{}
	}
	by := map[splitKey][]*layer{}
	var keys []splitKey
	for _, l := range g.layers {
		if l == nil || !l.ok || (l.pg != nil && !l.pg.in) || !g.gateUp2Shape(l) {
			continue
		}
		key := splitKey{l.mvu.q, l.mvu.rows, l.mvu.k}
		if by[key] == nil {
			keys = append(keys, key)
		}
		if len(by[key]) < 16 {
			by[key] = append(by[key], l)
		}
	}
	changed := false
	for _, key := range keys {
		ck := g.tuneKey(fmt.Sprintf("gateup2-%v-%dx%d", key.q, key.rows, key.k))
		old, had := g.gu2Win[key]
		if v, ok := lookupCache(g.kb.cacheDir, "gpugateup2", ck); ok && g.kb.tune != TuneForce {
			g.gu2Win[key] = v == "1"
		} else {
			r, ok := g.timeGateUp2(by[key])
			if !ok {
				continue
			}
			win := r < gateUp2Margin
			g.gu2Win[key] = win
			if backend.Verbose() {
				fmt.Fprintf(os.Stderr, "tier: Gate2 %v %dx%d: %.4f of the two launches over %d blocks: %v\n",
					key.q, key.rows, key.k, r, len(by[key]), win)
			}
			if ck != "" {
				v := "0"
				if win {
					v = "1"
				}
				saveCache(g.kb.cacheDir, "gpugateup2", ck, v)
			}
		}
		if !had || old != g.gu2Win[key] {
			changed = true
		}
	}
	if changed {
		g.dropGraph()
	}
}

// timeGateUp2 times the two arms over the blocks ls, interleaved, and returns
// the median of the per-round ratios Gate2/two-launch time; false where a
// launch failed or the trial did not resolve.
func (g *devTier) timeGateUp2(ls []*layer) (float64, bool) {
	tuneTrials.Add(1)
	bs := g.bs
	launch := func(s backend.Session, m mv, k backend.Kernel, bufs ...backend.Buf) error {
		w := 128
		if m.group {
			w = kernels.GroupSplitWidth(m.split)
		}
		n := m.rows * max(m.split, 1)
		return s.Launch(k, (n+w-1)/w, w, bufs...)
	}
	arm := func(s backend.Session, two bool) error {
		for _, l := range ls {
			if two {
				if err := launch(s, l.mvg, l.mvg.kern, l.gate.qs, l.gate.d, l.gate.sc, bs.a, bs.ax, bs.g); err != nil {
					return err
				}
				if err := launch(s, l.mvu, l.mvu.gated, l.up.qs, l.up.d, l.up.sc, bs.a, bs.ax, bs.act, bs.g); err != nil {
					return err
				}
				continue
			}
			if err := launch(s, l.mvu, l.mvu.gate2, l.up.qs, l.up.d, l.up.sc, bs.a, bs.ax, bs.act,
				l.gate.qs, l.gate.d, l.gate.sc); err != nil {
				return err
			}
		}
		return nil
	}
	timed := func(two bool) (float64, error) {
		const reps = 3
		t0 := time.Now()
		var err error
		g.dev.Session(func(s backend.Session) {
			backend.MarkTrial(s) // a profile tells a tuner's trial from the work
			for i := 0; i < reps && err == nil; i++ {
				err = arm(s, two)
			}
			if err == nil {
				err = s.Sync()
			}
		})
		return time.Since(t0).Seconds(), err
	}
	const rounds = 9
	var ratios []float64
	for round := 0; round < rounds+1; round++ {
		var a, b float64
		var ea, eb error
		if round%2 == 0 {
			a, ea = timed(true)
			b, eb = timed(false)
		} else {
			b, eb = timed(false)
			a, ea = timed(true)
		}
		if ea != nil || eb != nil {
			return 0, false
		}
		if round == 0 || a <= 0 {
			continue // the first round pays first-touch
		}
		ratios = append(ratios, b/a)
	}
	if len(ratios) == 0 {
		return 0, false // every round timed zero: a clock too coarse to resolve it
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	iqr := ratios[len(ratios)*3/4] - ratios[len(ratios)/4]
	return med, med > 0 && iqr/med <= splitTuneIQR
}

// gateUp2Swapped hands the Gate2 launch up's weight as the gate's and the
// gate's as up's: the violation the gate/up gate must see
// (SwapGateUp2ForTest, jitllmtest builds only).
var gateUp2Swapped bool
