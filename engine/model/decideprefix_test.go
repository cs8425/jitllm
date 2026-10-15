package model

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/tok/jinja"
)

// boardRequests is a game's run of decision requests, as examples/jivman
// asks them: a maze drawn as text, the same every request but for where
// Pac-Man stands and the pellets he has eaten, and one question about the
// junction he is at. Consecutive prompts share the system prompt and the
// board down to his row.
func boardRequests(t testing.TB, n int) [][]byte {
	t.Helper()
	var out [][]byte
	for r := range n {
		out = append(out, boardRequest(3+r, 35, '.')) // he walks right along row 35
	}
	return out
}

// boardRequest is one board with Pac-Man at (px, py), every open tile not
// yet eaten drawn as fill: a pellet, or for the violation a ghost.
func boardRequest(px, py int, fill byte) []byte {
	const w, h = 19, 41
	var rows []string
	for y := range h {
		var b strings.Builder
		for x := range w {
			switch {
			case y == 0 || y == h-1 || x == 0 || x == w-1 || (y%4 == 2 && x%4 == 2):
				b.WriteByte('#')
			case x == px && y == py:
				b.WriteByte('P')
			case y == py && x < px:
				b.WriteByte(' ') // eaten
			default:
				b.WriteByte(fill)
			}
		}
		rows = append(rows, `"`+b.String()+`"`)
	}
	return []byte(fmt.Sprintf(`{"state":{"maze":[%s],"legend":"# wall, . pellet, B ghost, P Pac-Man",`+
		`"pacman":{"x":%d,"y":%d}},"questions":{"move":{"type":"choice","instructions":"Which way should Pac-Man go?",`+
		`"criteria":{"up":"toward row 0","down":"toward the last row","left":"toward column 0","right":"toward the last column"}},`+
		`"safe":{"type":"noul","instructions":"Is Pac-Man out of danger?"}}}`, strings.Join(rows, ","), px, py))
}

func parseRequest(t testing.TB, b []byte) (jinja.Value, []DecisionQuestion) {
	t.Helper()
	v, err := jinja.FromJSON(b)
	if err != nil {
		t.Fatal(err)
	}
	state, _ := v.AsDict().Get("state")
	qv, _ := v.AsDict().Get("questions")
	qs, err := ParseDecisionQuestions(qv)
	if err != nil {
		t.Fatal(err)
	}
	return state, qs
}

// TestDecisionReusesThePrefix is the shared prefix cache on the decision
// models: a run of board requests asked of a Decider sharing its pages
// through a SharedStore (SetPromptCache) answers each exactly as a Decider
// without one does, bit for bit, while the counter says the shared prefix was
// restored rather than prefilled, on the host and with blocks on every
// device present, where the pages are named and mapped on the card and no
// position of history crosses the bus (Stats.KVPositionsMigrated).
// The violation restores the first board's pages for a later board, and the
// gate must see it (RULE 10).
func TestDecisionReusesThePrefix(t *testing.T) {
	reqs := boardRequests(t, 4)
	for _, c := range decisionCases {
		if !prefixReadout(c.kind) {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			m.SetKVF16(false)
			t.Run("host", func(t *testing.T) { decisionPrefixGate(t, m, nil, "", 0, reqs, 0) })
			for _, spec := range stepDevices() {
				g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer g.Close()
					t.Run("seam", func(t *testing.T) { decisionPrefixGate(t, m, g, spec, decisionBlocks(m)/2, reqs, c.tol) })
					t.Run("every-block", func(t *testing.T) { decisionPrefixGate(t, m, g, spec, -1, reqs, c.tol) })
				})
			}
		})
	}
}

// decisionPrefixGate holds a warm Decider's answers to a cold one's: bit for
// bit on the host (tol 0), and on a device within tol, the case's coherence
// bar: a device's prefill reduces in an order set by its chunk, and a
// restored prefix starts the rest of the prompt at another row. The violation
// must move an answer past tol.
// With every block on a device the violation is the card's own: a write
// into a page the cache names lands in it in place (tier.Config.KVCowFault),
// under every sequence that maps it.
func decisionPrefixGate(t *testing.T, m *Model, g *tier.GPU, spec string, n int, reqs [][]byte, tol float64) {
	open := func(store *SharedStore) *Decider {
		d, err := m.NewDecider(4096)
		if err != nil {
			t.Fatal(err)
		}
		if g != nil {
			if err := d.SetDevice(g, n); err != nil {
				t.Fatal(err)
			}
			if d.DeviceBlocks() == 0 {
				t.Fatalf("%s took no block: %s", g.Name(), g.Err())
			}
		}
		if store != nil {
			on, err := d.SetPromptCache(store, "decide-prefix-test")
			if err != nil {
				t.Fatal(err)
			}
			if !on {
				t.Fatal("the Decider did not take the shared store")
			}
		}
		return d
	}
	cold := open(nil)
	defer cold.Close()
	warm := open(NewSharedStore())
	defer warm.Close()
	var want [][]DecisionAnswer
	for _, b := range reqs {
		state, qs := parseRequest(t, b)
		a, err := cold.Decide(state, qs, nil)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, cloneAnswers(a))
	}
	if cold.PromptRestored() != 0 {
		t.Fatalf("a Decider with no store restored %d positions", cold.PromptRestored())
	}
	var st0 tier.Stats
	if g != nil {
		st0 = g.Stats()
	}
	for i, b := range reqs {
		state, qs := parseRequest(t, b)
		r0 := warm.PromptRestored()
		a, err := warm.Decide(state, qs, nil)
		if err != nil {
			t.Fatal(err)
		}
		if d := answersDiffer(a, want[i], tol); d != "" {
			t.Fatalf("request %d with the prefix cache: %s", i, d)
		}
		// Restored by aliasing the store's pages: what the session reads of
		// its history is held once, in the store, not copied into it.
		if i > 0 && g == nil && warm.st.KVSharedBytes() == 0 {
			t.Fatalf("request %d aliases no shared page", i)
		}
		if i > 0 && warm.PromptRestored() == r0 {
			t.Fatalf("request %d restored nothing: its prompt shares the board above row 35 with the last", i)
		}
	}
	t.Logf("%d prompt positions restored over %d requests", warm.PromptRestored(), len(reqs))
	if g != nil {
		st := g.Stats()
		mig, mapped := st.KVPositionsMigrated-st0.KVPositionsMigrated, st.KVPagesMapped-st0.KVPagesMapped
		t.Logf("%s: %d page(s) mapped on the card, %d copied on a write, %d position(s) migrated",
			g.Name(), mapped, st.KVPagesCopied-st0.KVPagesCopied, mig)
		if mig != 0 {
			t.Fatalf("%d positions of history crossed the bus for a warm prefix", mig)
		}
		if n < 0 && mapped == 0 {
			t.Fatal("no page was mapped on the card: the device restored nothing")
		}
	}

	// Every block on the card: the card's own violation. A seam keeps the
	// host's, which names the device's pages by the same stale keys.
	if g != nil && n < 0 {
		devicePrefixViolation(t, m, spec, n, reqs, want, tol)
		return
	}
	// The violation: every board after the first restores the first board's
	// pages, as a cache that named a page by less than the tokens before it
	// would. Its answers must move.
	stale := open(NewSharedStore())
	defer stale.Close()
	var first []int32
	staleKeyFault = func(key []int32) []int32 {
		if first == nil {
			first = append([]int32(nil), key...)
			return key
		}
		return first[:min(len(first), len(key))]
	}
	defer func() { staleKeyFault = nil }()
	// The first prompts it names are a board of ghosts', which every later
	// board then restores in place of its own.
	state, qs := parseRequest(t, boardRequest(3, 35, 'B'))
	if _, err := stale.Decide(state, qs, nil); err != nil {
		t.Fatal(err)
	}
	moved := false
	for i, b := range reqs {
		state, qs := parseRequest(t, b)
		a, err := stale.Decide(state, qs, nil)
		if err != nil {
			t.Fatal(err)
		}
		if answersDiffer(a, want[i], tol) != "" {
			moved = true
		}
	}
	t.Logf("the violation restored %d positions", stale.PromptRestored())
	if !moved {
		t.Fatal("restoring a stale board moved no answer: the gate cannot see a wrong prefix")
	}
}

// devicePrefixViolation runs the board requests through the shared cache on
// a tier whose writes land in a named page in place, and holds that its
// answers move.
func devicePrefixViolation(t *testing.T, m *Model, spec string, n int, reqs [][]byte, want [][]DecisionAnswer, tol float64) {
	gv, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.KVCowFault = true })}, testTierOpts(t)...)...)
	if err != nil || gv == nil {
		t.Fatalf("%s: the violation's tier: %v", spec, err)
	}
	defer gv.Close()
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if err := d.SetDevice(gv, n); err != nil {
		t.Fatal(err)
	}
	if on, err := d.SetPromptCache(NewSharedStore(), "decide-prefix-cow"); err != nil || !on {
		t.Fatalf("the violation's Decider did not take the store: %v", err)
	}
	// The first board, a board of ghosts that parts from it inside its pages
	// and writes there, and the first again, which maps its own pages back:
	// in place, they hold the ghosts' rows.
	moved := false
	for i, b := range [][]byte{reqs[0], boardRequest(3, 35, 'B'), reqs[0]} {
		state, qs := parseRequest(t, b)
		a, err := d.Decide(state, qs, nil)
		if err != nil {
			t.Fatal(err)
		}
		if i == 2 && answersDiffer(a, want[0], tol) != "" {
			moved = true
		}
	}
	t.Logf("the violation (writes into shared pages in place) restored %d positions", d.PromptRestored())
	if !moved {
		t.Fatal("writing into a shared page in place moved no answer: the gate cannot see it")
	}
}

func cloneAnswers(a []DecisionAnswer) []DecisionAnswer {
	out := make([]DecisionAnswer, len(a))
	for i, x := range a {
		x.Probs = append([]float64(nil), x.Probs...)
		out[i] = x
	}
	return out
}

// answersDiffer names the first probability of two requests' answers more
// than tol apart (tol 0: any bit), or "" when there is none.
func answersDiffer(a, b []DecisionAnswer, tol float64) string {
	if len(a) != len(b) {
		return fmt.Sprintf("%d answers, want %d", len(a), len(b))
	}
	for i := range a {
		if len(a[i].Probs) != len(b[i].Probs) {
			return fmt.Sprintf("answer %d: %d probabilities, want %d", i, len(a[i].Probs), len(b[i].Probs))
		}
		for j := range a[i].Probs {
			x, y := a[i].Probs[j], b[i].Probs[j]
			if (tol == 0 && math.Float64bits(x) != math.Float64bits(y)) || math.Abs(x-y) > tol || math.IsNaN(x) {
				return fmt.Sprintf("answer %d option %d: %v, want %v", i, j, a[i].Probs[j], b[i].Probs[j])
			}
		}
	}
	return ""
}

// TestDecisionWithAPrefixDoesNotAllocate is the allocation principle on a
// Decider pointed at the prefix cache: once warm, a run of board prompts
// that restore their shared prefix, prefill the rest and seal the restore
// points they pass makes zero engine heap allocations.
func TestDecisionWithAPrefixDoesNotAllocate(t *testing.T) {
	reqs := boardRequests(t, 3)
	for _, c := range decisionCases {
		if !prefixReadout(c.kind) {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			m.SetKVF16(false)
			t.Run("host", func(t *testing.T) { prefixAllocs(t, m, nil, reqs) })
			for _, spec := range stepDevices() {
				g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer g.Close()
					prefixAllocs(t, m, g, reqs)
				})
			}
		})
	}
}

// prefixAllocs holds a warm pass of board prompts through the shared store,
// every block on g when there is one, to zero engine allocations.
func prefixAllocs(t *testing.T, m *Model, g *tier.GPU, reqs [][]byte) {
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if g != nil {
		if err := d.SetDevice(g, -1); err != nil {
			t.Fatal(err)
		}
		if d.DeviceBlocks() == 0 {
			t.Fatalf("%s took no block: %s", g.Name(), g.Err())
		}
	}
	if on, err := d.SetPromptCache(NewSharedStore(), "decide-prefix-alloc"); err != nil || !on {
		t.Fatalf("the Decider did not take the shared store: %v", err)
	}
	var ps []decisionPrompt
	for _, b := range reqs {
		state, qs := parseRequest(t, b)
		for i := range qs {
			ids, groups, err := d.prompt(state, &qs[i], 0)
			if err != nil {
				t.Fatal(err)
			}
			ps = append(ps, decisionPrompt{ids: ids, groups: groups})
		}
	}
	pass := func() {
		for _, p := range ps {
			if _, err := d.forward(p.ids, p.groups, 0); err != nil {
				t.Fatal(err)
			}
		}
	}
	pass()
	for i := 0; i < 6 && countAllocs(pass).engine > 0; i++ {
	}
	r0, rd0 := d.PromptRestored(), m.PagerReads()
	w := countAllocs(pass)
	if d.PromptRestored() == r0 {
		t.Fatal("the window restored nothing: it measured cold prompts")
	}
	allocVerdict(t, fmt.Sprintf("%d prompt(s), %d positions restored", len(ps), d.PromptRestored()-r0),
		w, len(ps), m.PagerReads()-rd0, 0)
}

// prefixReadout says a readout's prompts go through the prefix cache: every
// decoder readout but Clef, which keeps every row's final residual. An
// encoder (Laya, d1-omni) attends both ways, so no prefix of its rows is a
// prefix's history.
func prefixReadout(k jlm.DecisionKind) bool {
	return k != jlm.DecisionLaya && k != jlm.DecisionD1Omni && k != jlm.DecisionClef
}
