package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestQwen4ExpOnEveryDevice runs each fixture with every block and the head
// on each device present against the host -- the hyper-connections' mixes
// and placements, the sigmoid-gated delta rule over a state slot that carries
// the n-gram tail, the n-gram block, QSA's pooled-key selection and the
// mixture (tier's q4e.go and qsa.go) -- on decode and on a prefill chunk.
// Then each violation alone, which the device must run rather than decline.
func TestQwen4ExpOnEveryDevice(t *testing.T) {
	for _, fx := range q4eFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openQ4E(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, q4eViolations, 1e-5)
		})
	}
}

// TestQwen4ExpDeviceRunsTheChunkWhole is the device gate's selection check: a
// prompt goes to the device as one batched submission -- every mix, the
// n-gram block and both QSA selections -- and a decode token runs them all
// there too. Without it, a block declined or a chunk replayed a row at a time
// would pass TestQwen4ExpOnEveryDevice as well.
func TestQwen4ExpDeviceRunsTheChunkWhole(t *testing.T) {
	m, g := openQ4E(t, "synth-qwen4exp")
	defer m.Close()
	c := m.Cfg
	nl, qsa := c.NLayer, 0
	for li := 0; li < nl; li++ {
		if c.QSAAt(li) {
			qsa++
		}
	}
	ran := 0
	for _, spec := range stepDevices() {
		// A replayed graph issues a token's launches without emitting them,
		// and the counters count emits: graphs off, so every call counts.
		gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) { c.NoGraph = true })},
			testTierOpts(t)...)...)
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(g.IDs) + 2)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if st.GPULayers() != nl {
				t.Fatalf("placed %d of %d blocks: %v %v", st.GPULayers(), nl, st.DeviceDeclines(), gpu.Err())
			}
			s0 := gpu.Stats()
			lg, err := st.Prefill(g.IDs)
			if err != nil {
				t.Fatal(err)
			}
			s1 := gpu.Stats()
			if _, err := st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			s2 := gpu.Stats()
			t.Logf("prefill of %d: %d mixes, %d n-gram, %d selections; a decode token: %d, %d, %d",
				len(g.IDs), s1.Q4EMixes-s0.Q4EMixes, s1.Q4ENGram-s0.Q4ENGram, s1.QSASelects-s0.QSASelects,
				s2.Q4EMixes-s1.Q4EMixes, s2.Q4ENGram-s1.Q4ENGram, s2.QSASelects-s1.QSASelects)
			for _, d := range [][2]tier.Stats{{s0, s1}, {s1, s2}} {
				if d[1].Q4EMixes-d[0].Q4EMixes != 2*nl || d[1].Q4ENGram-d[0].Q4ENGram != 1 ||
					d[1].QSASelects-d[0].QSASelects != qsa {
					t.Fatalf("a call did not run every block's mixes, the n-gram block and every selection " +
						"on the device in one submission")
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
