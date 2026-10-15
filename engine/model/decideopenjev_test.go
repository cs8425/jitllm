package model

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOpenJevPicturesMatchLlamaCpp holds openjev's picture request -- a
// chat-message state carrying its picture -- to llama-server's answer on the
// same GGUF and mmproj (scripts/decisiongold.sh): the picture's rows from the
// qwen3vl tower between its markers, and the letters read after them.
// openjev refuses a null state, as llama.cpp does, so pictures.json is not
// its request.
//
// The test model's tower is 144 wide, not a whole number of Q8_0 blocks: its
// matrices are padded with zero columns at conversion and every product pads
// its rows' tails as zeros (nn's ragged quantizer), which the gate counts
// (TestRaggedTowerProductsMatchTheOracle holds the products themselves).
func TestOpenJevPicturesMatchLlamaCpp(t *testing.T) {
	text, proj := testmodels.Path("openjev/tinyopenjev-for-testing-Q8_0.gguf"), testmodels.Path("openjev/mmproj-tinyopenjev-for-testing-Q8_0.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", text)
	}
	body, err := os.ReadFile(filepath.Join("testdata", "decision", "chatpicture.json"))
	if err != nil {
		t.Fatal(err)
	}
	req, err := ParseDecisionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Images) == 0 {
		t.Fatal("the request carries no picture, so this gate proves nothing")
	}
	imgs := make([]image.Image, len(req.Images))
	for i, u := range req.Images {
		if imgs[i], err = DecodeImageDataURL(u); err != nil {
			t.Fatal(err)
		}
	}
	gold := readDecisionGold(t, "tinyopenjev-for-testing-Q8_0.chatpicture.llamacpp.json")
	decide := func() ([]DecisionAnswer, int, int) {
		m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if m.Tower() == nil || m.Decision() != jlm.DecisionOpenJev {
			t.Fatalf("the container carries tower %v and readout %v", m.Tower() != nil, m.Decision())
		}
		ragged0 := nn.RaggedRows.Load()
		d, err := m.NewDecider(4096)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		ans, err := d.Decide(req.State, req.Questions, imgs)
		if err != nil {
			t.Fatal(err)
		}
		return ans, d.InputTokens(), int(nn.RaggedRows.Load() - ragged0)
	}
	ans, tokens, padded := decide()
	if tokens != gold.Usage.InputTokens {
		t.Errorf("ran %d positions, llama.cpp %d: the prompts differ", tokens, gold.Usage.InputTokens)
	}
	if padded == 0 {
		t.Fatal("no row had a ragged tail quantized: the ragged tower's products did not run at the packed width")
	}
	t.Logf("%d rows quantized with a ragged tail", padded)
	checkDecisions(t, req.Questions, ans, gold, 0.01, true)
}
