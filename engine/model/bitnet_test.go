package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The BitNet b1.58 fixtures (scripts/bitnetgold.py): transformers' own
// BitNetForCausalLM run through its "bitnet" quantizer, so every projection's
// input is int8 per token and every weight ternary, on both sides.
//
//	synth-bitnet-hf         the release's form (uint8 codes, a bf16
//	                        weight_scale), from safetensors: the container
//	                        holds every weight exactly, so the two arms part
//	                        only where an activation rounds across a boundary
//	                        one way and not the other
//	synth-bitnet-f16.gguf   bitnet.cpp's converter over the latent form, its
//	                        ternarization stored as F16: every block's
//	                        magnitude rounded to an f16 on the way, held to
//	                        the latent form's own golden
var bitnetFixtures = []struct {
	model, golden string
	bound         float64
}{
	{"synth-bitnet-hf", "synth-bitnet", bitnetNMSE},
	{"synth-bitnet-f16.gguf", "synth-bitnet-online", bitnetF16NMSE},
}

// bitnetNMSE is the bound against transformers where the container holds the
// weights exactly: the engine and the reference quantize the same activations
// with the same per-token scale, and part where a value sits on a rounding
// boundary (the reference divides by its scale, the engine multiplies by its
// reciprocal, and the two round a half differently). Measured in the 1e-7s;
// every violation in bitnetViolations reads 1e-3 or far above.
const bitnetNMSE = 1e-5

// bitnetF16NMSE is the same bound where the converter stored each block's
// magnitude as an f16 (up to 2^-12 relative per tensor, every tensor its own
// rounding).
const bitnetF16NMSE = 1e-4

// openBitNet opens a BitNet fixture -- a HuggingFace directory or a GGUF --
// and its golden.
func openBitNet(t *testing.T, model, golden string) (*Model, *c6Golden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "bitnet", golden+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &c6Golden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	var path string
	if filepath.Ext(model) == ".gguf" {
		p, ok := existingModel(testmodels.Path(model))
		if !ok {
			testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(model)+
				" (set JITLLM_MODELS to the model directory) -- run scripts/bitnetgold.py (RULE 11)")
		}
		path = jlmOf(t, p)
	} else {
		path = hfContainer(t, model)
	}
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	if !bitnetSelected(m) {
		m.Close()
		t.Fatalf("%s does not carry BitNet's graph: %+v", model, m.Cfg)
	}
	return m, g
}

// bitnetSelected reports whether the container is the graph the fixture is
// for: every projection ternary, both inner norms on every block, relu2 on
// the gate, NEOX rotary, per-token activations, the session's window as wide
// as the widest row.
func bitnetSelected(m *Model) bool {
	c := m.Cfg
	if c.Arch != "bitnet" || !c.TokenAct || c.Act != nn.ActReLU2 || !c.RopeNeox || !c.TiedEmbd {
		return false
	}
	for i := range m.layers {
		l := &m.layers[i]
		if l.attnSubNorm == nil || l.ffnSubNorm == nil {
			return false
		}
		for _, w := range []tensor{l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down} {
			if w.typ != quant.TQ2_0 {
				return false
			}
		}
	}
	return true
}

// TestBitNetMatchesTransformers holds BitNet b1.58 to transformers' own class
// and quantizer on every entry point that attends: decode, prefill at every
// length and the batched path. The tokenizer half compares the container's
// tokenizer to the release's tokenizer.json.
func TestBitNetMatchesTransformers(t *testing.T) {
	for _, fx := range bitnetFixtures {
		t.Run(fx.model, func(t *testing.T) {
			m, g := openBitNet(t, fx.model, fx.golden)
			defer m.Close()
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
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > fx.bound {
					t.Errorf("%s pos %d: NMSE %.3e against transformers (bound %.0e)", what, p, nmse, fx.bound)
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

// bitnetViolations each take one of BitNet's features out of the engine's
// copy of the model. Every one must move the logits far past bitnetNMSE, or
// the fixture cannot see it and TestBitNetMatchesTransformers certifies
// nothing about it.
var bitnetViolations = []c6Violation{
	{"no attention inner norm", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.attnSubNorm = nil })
	}},
	{"no FFN inner norm", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.ffnSubNorm = nil })
	}},
	{"SiLU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActSiLU }) }},
	{"NORM rotary", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	{"activations per 32, not per token", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.TokenAct = false })
	}},
}

// bitnetScaleViolations rewrite the release form's weight_scale tensors
// before conversion: the converter must carry each scale, and the right way
// up, or the container reads as one of these.
var bitnetScaleViolations = []struct {
	name string
	f    func(name string, ws float32) float32
}{
	{"every weight's scale one", func(string, float32) float32 { return 1 }},
	{"wo's scale inverted", func(n string, ws float32) float32 {
		if strings.HasSuffix(n, "o_proj.weight_scale") {
			return 1 / ws
		}
		return ws
	}},
}

// TestBitNetFeaturesAreLoadBearing runs each violation against the release
// form's golden and demands it fail by a wide margin: a feature the fixture
// cannot see is a feature the match does not certify.
func TestBitNetFeaturesAreLoadBearing(t *testing.T) {
	m, g := openBitNet(t, bitnetFixtures[0].model, bitnetFixtures[0].golden)
	defer m.Close()
	worstOf := func(t *testing.T, m *Model) float64 {
		t.Helper()
		worst := 0.0
		st := m.NewState(len(g.IDs) + 1)
		defer st.Close()
		for p, id := range g.IDs {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			worst = max(worst, llama4Cmp(g.Pos[p].Head, lg))
		}
		return worst
	}
	check := func(t *testing.T, name string, worst float64) {
		t.Helper()
		if !(worst > 100*bitnetNMSE) {
			t.Errorf("violation %q reads NMSE %.3e: within 100x the bound %.0e, so the fixture cannot see it",
				name, worst, bitnetNMSE)
		}
		t.Logf("violation %q: worst NMSE %.3e", name, worst)
	}
	if w := worstOf(t, m); w > bitnetNMSE {
		t.Fatalf("the unviolated model reads %.3e: the violations below would measure nothing", w)
	}
	for _, v := range bitnetViolations {
		t.Run(v.name, func(t *testing.T) {
			undo := v.mut(m)
			defer undo()
			check(t, v.name, worstOf(t, m))
		})
	}
	for _, v := range bitnetScaleViolations {
		t.Run(v.name, func(t *testing.T) {
			dir := bitnetRescaled(t, testmodels.Path(bitnetFixtures[0].model), v.f)
			vm, err := Open(hfContainerAt(t, dir), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer vm.Close()
			check(t, v.name, worstOf(t, vm))
		})
	}
}

// bitnetRescaled copies a release-form directory with every weight_scale
// rewritten by f (bf16 in, bf16 out) and returns the copy.
func bitnetRescaled(t *testing.T, src string, f func(name string, ws float32) float32) string {
	t.Helper()
	dst := t.TempDir()
	ents, err := os.ReadDir(src)
	if err != nil {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+src+" -- run scripts/bitnetgold.py (RULE 11)")
	}
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if e.Name() == "model.safetensors" {
			n := binary.LittleEndian.Uint64(b)
			var hdr map[string]json.RawMessage
			if err := json.Unmarshal(b[8:8+n], &hdr); err != nil {
				t.Fatal(err)
			}
			changed := 0
			for name, raw := range hdr {
				if filepath.Ext(name) != ".weight_scale" {
					continue
				}
				var ti struct {
					DType   string   `json:"dtype"`
					Offsets [2]int64 `json:"data_offsets"`
				}
				if err := json.Unmarshal(raw, &ti); err != nil || ti.DType != "BF16" {
					t.Fatalf("%s: %s %v", name, ti.DType, err)
				}
				at := 8 + int64(n) + ti.Offsets[0]
				ws := math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[at:])) << 16)
				binary.LittleEndian.PutUint16(b[at:], uint16(math.Float32bits(bf16Of(f(name, ws)))>>16))
				changed++
			}
			if changed == 0 {
				t.Fatal("no weight_scale tensor in the fixture: the violation changes nothing")
			}
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

// bf16Of rounds x to the nearest bfloat16, ties to even.
func bf16Of(x float32) float32 {
	u := math.Float32bits(x)
	u += 0x7FFF + (u >> 16 & 1)
	return math.Float32frombits(u &^ 0xFFFF)
}

// hfContainerAt converts a HuggingFace directory into a container in the
// test's temporary directory.
func hfContainerAt(t *testing.T, dir string) string {
	t.Helper()
	dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
	if _, err := convert.FromSafetensors(dir, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("convert %s: %v", dir, err)
	}
	return dst
}
