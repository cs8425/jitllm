package model

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestClefPicturesMatchReference holds a Clef request carrying two pictures
// (pictures.json) to Clef's own code on the random fixture with a toy tower
// whose position table is the release's (clefgold.py fixture-vl, the release's Qwen3VLProcessor): the prompt row for
// row -- each picture's markers and rows after "STATE:\n", a newline, then the
// state -- every question's and option's span, and the head's logits, on the
// host and with every block on each device. The violation is the pictures
// left out, which must move the logits past what the gate reads clean.
func TestClefPicturesMatchReference(t *testing.T) {
	text := testmodels.Path(filepath.Join("clef", "fixture-vl", "clef-fixture-vl.gguf"))
	proj := testmodels.Path(filepath.Join("clef", "fixture-vl", "mmproj-clef-fixture-vl.gguf"))
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- run scripts/clefgold.py fixture-vl (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Tower() == nil || m.Decision() != jlm.DecisionClef {
		t.Fatal("the container carries no tower or no Clef head, so this gate proves nothing")
	}
	body, err := os.ReadFile(filepath.Join("testdata", "decision", "pictures.json"))
	if err != nil {
		t.Fatal(err)
	}
	_, state, qs, urls, err := ParseDecisionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	imgs := make([]image.Image, len(urls))
	for i, u := range urls {
		if imgs[i], err = DecodeImageDataURL(u); err != nil {
			t.Fatal(err)
		}
	}
	g := readClefGoldOf(t, "clef-fixture-vl", "pictures")
	run := func(t *testing.T, gpu *tier.GPU, pics []image.Image) (*clefPrompt, []float32) {
		d, err := m.NewDecider(4096)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		if gpu != nil {
			if err := d.SetDevice(gpu, -1); err != nil {
				t.Fatal(err)
			}
			if d.DeviceBlocks() != m.Cfg.NLayer {
				t.Fatalf("the device took %d of %d blocks (%s)", d.DeviceBlocks(), m.Cfg.NLayer, gpu.Err())
			}
		}
		if err := d.Prepare(state, qs, len(pics)); err != nil {
			t.Fatal(err)
		}
		if len(pics) > 0 {
			if d.pics, err = d.cutPictures(pics); err != nil {
				t.Fatal(err)
			}
		}
		p, err := d.clefPrompt(state, qs)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(p.opts))
		if err := d.clefForward(p, out); err != nil {
			t.Fatal(err)
		}
		return p, out
	}
	// The tower's activations are q8 here (the engine's arithmetic for every
	// tower), so the gate is held to the reference with its tower's inputs
	// rounded the same way (logits_a8); the f32 reference is logged.
	if len(g.LogitsA8) == 0 {
		t.Fatal("the golden has no logits_a8: regenerate it with scripts/clefgold.py fixture-vl")
	}
	a8 := g
	a8.Logits = g.LogitsA8
	check := func(t *testing.T, gpu *tier.GPU) (float64, float64) {
		p, got := run(t, gpu, imgs)
		if len(p.ids) != len(g.IDs) {
			t.Fatalf("the prompt is %d rows, the reference's %d", len(p.ids), len(g.IDs))
		}
		for i := range p.ids {
			if p.ids[i] != g.IDs[i] {
				t.Fatalf("row %d is %d, the reference's %d", i, p.ids[i], g.IDs[i])
			}
		}
		at := 0
		for qi, q := range g.Questions {
			if s := p.qs[qi]; s.lo != q.Span[0] || s.hi != q.Span[1] {
				t.Errorf("%s: span [%d,%d), the reference's %v", q.ID, s.lo, s.hi, q.Span)
			}
			for _, o := range q.Options {
				if s := p.opts[at]; s.lo != o[0] || s.hi != o[1] {
					t.Errorf("%s option %d: span [%d,%d), the reference's %v", q.ID, at, s.lo, s.hi, o)
				}
				at++
			}
		}
		return clefWorst(t, a8, got), clefWorst(t, g, got)
	}
	clean, f32 := check(t, nil)
	t.Logf("host: %d pictures, worst logit difference from Clef's own code %.2e at the engine's arithmetic, "+
		"%.2e against its f32 tower", len(imgs), clean, f32)
	// The bound: both sides round every tower activation to q8, from f32
	// values summed in different orders, so a value at a rounding edge goes
	// a step apart; the text-only fixture reads 1.7e-6 and the pictures
	// 5.5e-4, a sixtieth of what leaving them out moves.
	const clefPictureTol = 2e-3
	if clean > clefPictureTol {
		t.Fatalf("host: worst logit %.2e from the reference, tolerance %.0e", clean, clefPictureTol)
	}
	// The violation: the same request with no pictures, held to the
	// picture golden's logits where the counts allow.
	_, bare := run(t, nil, nil)
	if w := clefWorstLoose(g, bare); !(w > clefSeen) {
		t.Errorf("the pictures left out move the logits by %.2e: the gate cannot see them", w)
	} else {
		t.Logf("violation, the pictures left out: %.2e", w)
	}
	for _, spec := range stepDevices() {
		gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			w, _ := check(t, gpu)
			t.Logf("%s, every block: worst logit %.2e from the reference", spec, w)
			if w > clefPictureTol {
				t.Fatalf("%s: worst logit %.2e", spec, w)
			}
		})
	}
}
