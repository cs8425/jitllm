package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// moveDevices is the two devices of one tier the device-to-device gates move
// blocks between: JITLLM_MOVE_DEVICES, or the laptop's one card under two
// backends.
func moveDevices() (string, string) {
	spec := os.Getenv("JITLLM_MOVE_DEVICES")
	if spec == "" {
		spec = "cuda:0,vulkan:1"
	}
	a, b, _ := strings.Cut(spec, ",")
	return a, b
}

// TestBlocksMoveBetweenDevices moves every block of a sequence from one device
// of a tier to another mid-generation and back (MoveBlocks), teacher-forced
// on an unmoved control's tokens: no prompt row is computed again, every block
// lands where it was sent, and the logits after each move are the control's --
// bit for bit between two devices running the same kernels, and inside the
// reduction-order band between two backends (RULE 11c).
//
// The upper arm moves only the upper half, so the output projection stays
// behind on a device that no longer holds the last block: the projection's
// own submission, which no placement reached before a block could be moved.
func TestBlocksMoveBetweenDevices(t *testing.T) {
	names := append([]string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf"}, relocModels()...)
	for _, name := range names {
		t.Run(name, func(t *testing.T) { blocksMoveBetweenDevices(t, jlmOf(t, testmodels.Path(name)), false) })
		t.Run(name+"/upper", func(t *testing.T) { blocksMoveBetweenDevices(t, jlmOf(t, testmodels.Path(name)), true) })
	}
}

func blocksMoveBetweenDevices(t *testing.T, path string, upper bool, opts ...Option) {
	a, b := moveDevices()
	g, err := tier.OpenWith(tier.WithDevices(a+","+b), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	m, err := Open(path, append(opts, WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}))...)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const ntok, moveAt, backAt = 12, 4, 8
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	maxSeq := len(prompt) + ntok + 1

	// run feeds ids (nil: its own greedy picks) and moves the blocks at
	// moveAt and back at backAt when move is set.
	run := func(ids []int32, move bool) ([]int32, [][]float32, []BlockMove) {
		st := m.NewState(maxSeq)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		placed := st.placedOn(a)
		if len(placed) == 0 {
			t.Skipf("%s took no block of this model: %s", a, g.Err())
		}
		if upper {
			if len(placed) < 2 {
				t.Skipf("%s took %d block(s); the upper half of one is all of it", a, len(placed))
			}
			placed = placed[len(placed)/2:]
		}
		logits, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		var lg [][]float32
		var moves []BlockMove
		for i := 0; i < ntok; i++ {
			lg = append(lg, append([]float32(nil), logits...))
			next := Greedy(logits)
			if ids != nil {
				next = ids[i]
			}
			out = append(out, next)
			if move && (i == moveAt || i == backAt) {
				to, from := b, a
				if i == backAt {
					to, from = a, b
				}
				rows := st.PrefillRows()
				mv := st.moveAll(t, from, to, placed)
				if got := st.PrefillRows() - rows; got != 0 {
					t.Fatalf("the move %s -> %s computed %d prompt rows again", from, to, got)
				}
				for _, li := range placed {
					if at, _ := st.DeviceOf(li); at != to {
						t.Fatalf("block %d is on %q after a move to %s (declined %v)", li, at, to, mv.Declined)
					}
				}
				moves = append(moves, mv)
			}
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return out, lg, moves
	}

	ctl, ctlLogits, _ := run(nil, false)
	_, movedLogits, moves := run(ctl, true)
	if len(moves) != 2 || len(moves[0].Moved) == 0 || moves[0].KVBytes+moves[0].RecBytes == 0 {
		t.Fatalf("the moves did not run: %+v", moves)
	}
	t.Logf("%s <-> %s: %d block(s), %d KV and %d recurrent bytes, path %s, head moved %v",
		a, b, len(moves[0].Moved), moves[0].KVBytes, moves[0].RecBytes, moves[0].Path, moves[0].Head)
	same := strings.SplitN(a, ":", 2)[0] == strings.SplitN(b, ":", 2)[0]
	var band float64
	defer func() { t.Logf("largest logit difference after a move: %g", band) }()
	for i := range ctlLogits {
		var worst float64
		for j, v := range ctlLogits[i] {
			worst = max(worst, math.Abs(float64(v-movedLogits[i][j])))
		}
		if math.IsNaN(worst) {
			t.Fatalf("token %d: a non-finite logit", i)
		}
		band = max(band, worst)
		switch {
		case i <= moveAt && worst != 0:
			t.Fatalf("token %d is before any move and its logits differ by %g", i, worst)
		case same && worst != 0:
			t.Fatalf("token %d: two devices running the same kernels differ by %g after a move", i, worst)
		case worst > 2.5:
			t.Fatalf("token %d: the moved logits are %g from the control's, outside the band", i, worst)
		}
	}
}

// placedOn lists the blocks the named device holds for this State.
func (s *State) placedOn(name string) []int {
	var out []int
	for li := s.lo; li < s.hi; li++ {
		if at, ok := s.DeviceOf(li); ok && at == nn.DeviceName(name) {
			out = append(out, li)
		}
	}
	return out
}

// moveAll moves the blocks to the named device, a KV-sharing group whole,
// failing the test on an error.
func (s *State) moveAll(t *testing.T, from, to string, blocks []int) BlockMove {
	t.Helper()
	var all BlockMove
	done := map[int]bool{}
	for _, li := range blocks {
		lo, hi := s.kvSpan(li)
		if done[lo] {
			continue
		}
		done[lo] = true
		mv, err := s.MoveBlocks(lo, hi, to)
		if err != nil {
			t.Fatalf("%s -> %s, blocks %d..%d: %v", from, to, lo, hi-1, err)
		}
		all.Moved = append(all.Moved, mv.Moved...)
		all.Declined = append(all.Declined, mv.Declined...)
		all.KVBytes += mv.KVBytes
		all.RecBytes += mv.RecBytes
		all.Path, all.Head = mv.Path, all.Head || mv.Head
	}
	return all
}
