package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The gemma3 and internvl towers on every device backend: every block placed,
// one block held to the host at the bound TestTowerOnDeviceMatchesHost sets,
// the whole tower's output held to the host's, and the blocks brought home
// again with the host's answer unchanged (relocation).

func nmseOf(got, want []float32) (nmse, worst float64) {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
		worst = math.Max(worst, math.Abs(d))
	}
	return num / den, worst
}

func towerOnDevices(t *testing.T, open func(testing.TB, ...Option) (*Model, *Tower)) {
	m, tw := open(t)
	defer m.Close()
	c := tw.Cfg
	px := rainbow(c.ImageSz)
	// Normalised as the tower's own preprocessing would, so the residual has
	// the magnitude a real picture gives it.
	for i := range px {
		ch := i % 3
		px[i] = float32((float64(px[i]) - c.Mean[ch]) / c.Std[ch])
	}
	encode := func(s *State) (after0, out []float32) {
		s.vis.afterBlock = func(li int, x []float32) {
			if li == 0 && after0 == nil {
				after0 = append([]float32(nil), x...)
			}
		}
		e, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return after0, append([]float32(nil), e...)
	}
	host := tw.testState()
	want0, want := encode(host)
	host.Close()

	ran := 0
	for _, spec := range stepDevices() {
		g, err := tier.OpenWith(tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			// Every block, and the whole tower's output. A card whose free
			// memory does not hold the tower resident takes it paged: the
			// blocks stream through the device (paging is the design), and
			// the answer is held to the host's the same way.
			all := tw.testState()
			all.SetDevice(g)
			if n := all.GPUBlocks(); n != c.NLayer {
				why := g.Err()
				all.Close()
				pm, ptw := open(t, WithPlacement(Placement{Rest: &Place{On: spec, Stream: true}}))
				defer pm.Close()
				all = ptw.testState()
				all.SetDevice(g)
				if n2 := all.GPUBlocks(); n2 != c.NLayer {
					all.Close()
					t.Fatalf("%s took %d of %d tower blocks resident (%s) and %d streamed (%s)",
						spec, n, c.NLayer, why, n2, g.Err())
				}
				t.Logf("%s held %d of %d tower blocks resident (%s): the tower streams through it", spec, n, c.NLayer, why)
			}
			s0 := g.Stats().Submits
			_, gotAll := encode(all)
			if g.Stats().Submits == s0 {
				t.Fatalf("%s: the encode submitted nothing to the device: the tower ran on the host", spec)
			}
			nm, worst := nmseOf(gotAll, want)
			t.Logf("%s, all %d blocks: projector output NMSE %.3e, max|d| %.4f", spec, c.NLayer, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 1e-2 {
				t.Errorf("%s: the whole tower on the device reads NMSE %.3e against the host", spec, nm)
			}
			// Relocation: the blocks come home through the one placement path
			// (State.SetDevice releases them) and the host's answer is the
			// host's again, bit for bit.
			if err := all.SetDevice(nil); err != nil {
				t.Fatalf("%s: bringing the tower home: %v", spec, err)
			}
			_, back := encode(all)
			all.Close()
			if nb, _ := nmseOf(back, want); nb != 0 {
				t.Errorf("%s: after the blocks came home the host reads NMSE %.3e against itself", spec, nb)
			}

			// One block, held to the host after it.
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != 1 {
				t.Fatalf("asked for 1 device block, got %d", s.GPUBlocks())
			}
			s1 := g.Stats().Submits
			got0, _ := encode(s)
			if g.Stats().Submits == s1 {
				t.Fatalf("%s: block 0's encode submitted nothing to the device", spec)
			}
			nm, worst = nmseOf(got0, want0)
			t.Logf("%s, after block 0: NMSE %.3e, max|d| %.4f", spec, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || math.IsInf(worst, 0) {
				t.Fatalf("NMSE %v on %s: the comparison is degenerate", nm, spec)
			}
			bound := 5e-5
			if st := g.Stats(); st.VoltaGemm+st.GemmF16 > 0 {
				bound = 5e-4
			}
			if nm > bound {
				t.Fatalf("NMSE %.3e after ONE block on %s (bound %.0e)", nm, spec, bound)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

func TestGemma3TowerOnDevice(t *testing.T)   { towerOnDevices(t, openGemma3) }
func TestInternVLTowerOnDevice(t *testing.T) { towerOnDevices(t, openInternVL) }

// TestGemma3TowerPagesThroughTheDevice runs the Gemma 3 tower placed to
// stream (Place.Stream) under budgets below what it holds resident: every
// block must still be placed, blocks must page in during the encode, and the
// projector output must be the resident tower's bit for bit -- a page-in
// uploads the same bytes. Before streamed tower blocks paged out to make room
// for a picture's row set (resizeTower), the set sent them home instead and
// the device ran a fraction of the tower.
func TestGemma3TowerPagesThroughTheDevice(t *testing.T) {
	m0, tw0 := openGemma3(t)
	px := rainbow(tw0.Cfg.ImageSz)
	m0.Close()
	ran := 0
	for _, spec := range stepDevices() {
		run := func(budget uint64) (out []float32, placed, ins int, used uint64, ok bool) {
			opts := []tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)}
			if budget > 0 {
				opts = append(opts, tier.WithBudget(budget))
			}
			g, err := tier.OpenWith(opts...)
			if err != nil || g == nil {
				return nil, 0, 0, 0, false
			}
			defer g.Close()
			m, tw := openGemma3(t, WithPlacement(Placement{Rest: &Place{On: spec, Stream: true}}))
			defer m.Close()
			s := tw.testState()
			defer s.Close()
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			used = g.Budgets()[0].Used
			p0 := g.Stats().PageIns
			e, err := s.Encode(px)
			if err != nil {
				t.Fatalf("%s under %d bytes: %v", spec, budget, err)
			}
			return append([]float32(nil), e...), s.GPUBlocks(), g.Stats().PageIns - p0, used, true
		}
		ref, placed, ins, used, ok := run(0)
		if !ok {
			t.Logf("%s: not present", spec)
			continue
		}
		ran++
		if placed != tw0.Cfg.NLayer || ins != 0 {
			t.Fatalf("%s: the unbounded arm placed %d of %d blocks with %d page-ins: it is not the resident tower",
				spec, placed, tw0.Cfg.NLayer, ins)
		}
		paged := 0
		for _, frac := range []float64{0.75, 0.6} {
			b := uint64(float64(used) * frac)
			got, p, in, _, _ := run(b)
			t.Logf("%s, %d of %d bytes: %d of %d blocks placed, %d page-ins", spec, b, used, p, tw0.Cfg.NLayer, in)
			if p != tw0.Cfg.NLayer || in == 0 {
				continue
			}
			paged++
			if nm, worst := nmseOf(got, ref); worst != 0 {
				t.Fatalf("%s under %d bytes: paging the tower changed its output (NMSE %.3e)", spec, b, nm)
			}
		}
		if paged == 0 {
			t.Fatalf("%s: no budget below the resident tower paged every block through the device", spec)
		}
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
