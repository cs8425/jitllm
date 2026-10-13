package model

import (
	"math"
	"slices"
	"testing"
)

const q4eGoldScript = "scripts/qwen4expgold.py"

func openQ4E(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "qwen4exp", q4eGoldScript, name)
}

// q4eFixtures names each fixture and what its container must carry: four
// hyper-connection streams at rank 16; linear, attention, linear, attention;
// QSA on both attention blocks (budget 8 at a ratio of 4, so from position 12
// a query reads a subset of its keys); the n-gram block on block 2 (3-grams,
// two heads each, rows of 32); the delta rule's gate a sigmoid.
var q4eFixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-qwen4exp", func(m *Model) bool {
		c := m.Cfg
		return c.Arch == "qwen4exp" && c.HCMult == 4 && c.HCLowRank == 16 && c.QSARatio == 4 &&
			c.IdxTopK == 8 && c.IdxHeads == 8 && c.IdxHeadDim == 32 && c.NGramBlock == 3 &&
			c.NGram == 3 && c.NGramHeadsPer == 2 && c.NGramRowDim == 32 && c.NGramConv == 4 &&
			c.DeltaSigmoidGate && c.NRot == 16 && c.RopeNeox && c.AttnOutGate &&
			c.LayerKind(0).Recurrent() && !c.LayerKind(1).Recurrent() &&
			c.LayerKind(2).Recurrent() && !c.LayerKind(3).Recurrent() && m.ngram != nil
	}},
}

// TestQwen4ExpMatchesTransformers holds each fixture to transformers' own
// Qwen4ExpForCausalLM on decode, prefill at every length and the batched
// path, and the container's tokenizer to Qwen3.8-Flash-Next's tokenizer.json.
func TestQwen4ExpMatchesTransformers(t *testing.T) {
	for _, fx := range q4eFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openQ4E(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if blocks := len(g.IDs) / m.Cfg.QSARatio; blocks <= m.Cfg.IdxTopK/m.Cfg.QSARatio {
				t.Fatalf("%d golden positions are %d QSA blocks against a budget of %d: every key is read",
					len(g.IDs), blocks, m.Cfg.IdxTopK/m.Cfg.QSARatio)
			}
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			for _, tk := range g.Texts {
				if got := m.Vocab.Encode(tk.Text, false); !slices.Equal(got, tk.IDs) {
					t.Errorf("Encode(%q)\n  ours           %v\n  tokenizer.json %v", tk.Text, got, tk.IDs)
				}
			}
			worst := c6Worst(t, m, g, func(what string, p int, lg []float32) float64 {
				t.Helper()
				nmse := llama4Cmp(g.Pos[p].Head, lg)
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > c6NMSE {
					t.Errorf("%s pos %d: NMSE %.3e against transformers (bound %.0e)", what, p, nmse, c6NMSE)
				}
				if am := Greedy(lg); am != g.Pos[p].Argmax {
					t.Errorf("%s pos %d: argmax %d, transformers %d", what, p, am, g.Pos[p].Argmax)
				}
				return nmse
			})
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}
