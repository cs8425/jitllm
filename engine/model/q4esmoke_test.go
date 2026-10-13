package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

func TestQ4ESmokeLogits(t *testing.T) {
	m, g := openQ4E(t, "synth-qwen4exp")
	defer m.Close()
	run := func(st *State) [][]float32 {
		var out [][]float32
		for _, id := range g.IDs {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		st.Reset()
		pl, err := st.Prefill(g.IDs)
		if err != nil {
			t.Fatal(err)
		}
		return append(out, append([]float32(nil), pl...))
	}
	st := m.NewState(len(g.IDs) + 8)
	want := run(st)
	st.Close()
	for _, spec := range stepDevices() {
		gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))
		if err != nil || gpu == nil {
			continue
		}
		st := m.NewState(len(g.IDs) + 8)
		st.SetDeviceLayers(gpu, -1)
		s0 := gpu.Stats()
		got := run(st)
		s1 := gpu.Stats()
		worst := 0.0
		for p := range want {
			var num, den float64
			for i := range want[p] {
				d := float64(got[p][i] - want[p][i])
				num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
			}
			worst = max(worst, num/den)
		}
		t.Logf("%s: placed %d, worst NMSE %.3g, mixes %d ngram %d, err %q", spec, st.GPULayers(), worst,
			s1.Q4EMixes-s0.Q4EMixes, s1.Q4ENGram-s0.Q4ENGram, gpu.Err())
		st.Close()
		gpu.Close()
	}
}
