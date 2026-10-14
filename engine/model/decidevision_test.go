package model

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/tok/jinja"
)

// d1-3B's decisions over pictures, against the checkpoint's own SystemOne
// (runner.py: cap_pixels, the processor, the hybrid stack) in float32 on the
// pictures scripts/lfm2visiongold.py ran: with a text state and with none,
// a choice, a noul and a score per request.

type d1VisionCase struct {
	Image     string
	State     *string
	Questions json.RawMessage
	Response  struct {
		Answers map[string]struct {
			Type          string
			Noul          float64
			Probabilities map[string]float64
		}
	}
}

func readD1VisionCases(t *testing.T) []d1VisionCase {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(d1Gold, "decisions.json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2visiongold.py (RULE 11)", err)
	}
	var cs []d1VisionCase
	if err := json.Unmarshal(b, &cs); err != nil {
		t.Fatal(err)
	}
	if len(cs) == 0 {
		t.Fatal("the golden holds no case")
	}
	return cs
}

// d1VisionWorst runs every case on d and returns the worst probability
// difference against the reference.
func d1VisionWorst(t *testing.T, d *Decider, cs []d1VisionCase) float64 {
	t.Helper()
	worst := 0.0
	for _, c := range cs {
		qv, err := jinja.FromJSON(c.Questions)
		if err != nil {
			t.Fatal(err)
		}
		qs, err := ParseDecisionQuestions(qv)
		if err != nil {
			t.Fatal(err)
		}
		state := jinja.None()
		if c.State != nil {
			state = jinja.NewString(*c.State)
		}
		img := decodeImage(t, filepath.Join(d1Gold, c.Image))
		ans, err := d.Decide(state, qs, []image.Image{img})
		if err != nil {
			t.Fatal(err)
		}
		for i, q := range qs {
			ref, ok := c.Response.Answers[q.ID]
			if !ok {
				t.Fatalf("the golden has no answer to %s", q.ID)
			}
			if ref.Type == "noul" {
				worst = math.Max(worst, math.Abs(ans[i].Noul-ref.Noul))
				continue
			}
			if len(ans[i].Probs) != len(q.Options) {
				t.Fatalf("%s: %d probabilities for %d options", q.ID, len(ans[i].Probs), len(q.Options))
			}
			for j, o := range q.Options {
				worst = math.Max(worst, math.Abs(ans[i].Probs[j]-ref.Probabilities[o.Key]))
			}
		}
		t.Logf("%s, state %v: %d questions, %d positions", c.Image, c.State != nil, len(qs), d.InputTokens())
	}
	return worst
}

// TestD1DecidesOverPicturesAsItsReference holds d1-3B's answers over each
// picture to its own code's, on the host and with every block on each
// device; a picture left out of the prompt is the violation.
func TestD1DecidesOverPicturesAsItsReference(t *testing.T) {
	m := openD1VL(t)
	defer m.Close()
	cs := readD1VisionCases(t)
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	// Q8_0 weights and int8 activations against the reference's float32,
	// through 27 tower blocks and 30 text blocks; the text-only gate reads
	// 0.021 against llama.cpp.
	const bound = 0.05
	worst := d1VisionWorst(t, d, cs)
	t.Logf("host: worst probability difference %.4f against d1's own code", worst)
	if math.IsNaN(worst) || worst > bound {
		t.Fatalf("host: worst probability difference %.4f (bound %g)", worst, bound)
	}

	// The violation: the same requests with the picture's rows replaced by
	// another picture's.
	other := make([]d1VisionCase, len(cs))
	for i, c := range cs {
		other[i] = c
		other[i].Image = cs[(i+2)%len(cs)].Image
	}
	if other[0].Image != cs[0].Image {
		bad := d1VisionWorst(t, d, other)
		t.Logf("violation, each request answered over another picture: worst %.4f", bad)
		if !(bad > 2*worst && bad > bound) {
			t.Errorf("another picture reads %.4f against a clean %.4f: the gate cannot see the picture", bad, worst)
		}
	}

	for _, spec := range stepDevices() {
		gpu, err := tier.OpenWith(tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			dd, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer dd.Close()
			if err := dd.SetDevice(gpu, -1); err != nil {
				t.Fatal(err)
			}
			// Every block where the card holds the model (the V100s); on a
			// smaller card the partial seam the bar asks for as well, the
			// count asserted and printed.
			n := dd.DeviceBlocks()
			t.Logf("%s holds %d of %d text blocks (%s)", spec, n, m.Cfg.NLayer, gpu.Err())
			if n == 0 {
				t.Fatalf("%s took no text block, so this arm ran on the host", spec)
			}
			w := d1VisionWorst(t, dd, cs)
			t.Logf("%s, every text block: worst %.4f against d1's own code", spec, w)
			if math.IsNaN(w) || w > bound {
				t.Fatal(fmt.Sprintf("%s: worst %.4f (bound %g)", spec, w, bound))
			}
		})
	}
}
