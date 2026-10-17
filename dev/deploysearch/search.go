package main

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

// Config is one deployment setting set: jitllmd serve flags added to the base
// arguments, on one named build. The control has none.
type Config struct {
	Name  string
	Bin   string
	Flags [][2]string // flag name (no dash), value, in a fixed order
}

// Args renders the flags for the command line.
func (c Config) Args() []string {
	var a []string
	for _, f := range c.Flags {
		a = append(a, "-"+f[0], f[1])
	}
	return a
}

// Metrics are one restart's measurements on one request mix.
type Metrics struct {
	TTFTp50, TTFTp99 float64 // ms
	ITLp99           float64 // ms
	Goodput          float64 // requests/s within the SLO
	AggRate          float64 // tokens/s
	Errors, Requests int
	LoadSec          float64 // process start to serving
	HostMiB, VRAMMiB float64 // peak resident host memory, peak device memory
}

// Mem is the memory the setting costs: host plus device.
func (m Metrics) Mem() float64 { return m.HostMiB + m.VRAMMiB }

// Sample is one restart of one config in one round.
type Sample struct {
	Config string
	Round  int
	M      Metrics
}

// Grid builds the one-factor-at-a-time candidates around the default plus any
// explicit bundles, deduplicated by name. knobs maps a flag to its values;
// order fixes the candidate order.
func Grid(bin string, order []string, knobs map[string][]string, combos []string) ([]Config, error) {
	out := []Config{{Name: "default", Bin: bin}}
	seen := map[string]bool{"default": true}
	add := func(fl [][2]string) {
		parts := make([]string, len(fl))
		for i, f := range fl {
			parts[i] = f[0] + "=" + f[1]
		}
		n := strings.Join(parts, "+")
		if !seen[n] {
			seen[n] = true
			out = append(out, Config{Name: n, Bin: bin, Flags: fl})
		}
	}
	for _, k := range order {
		for _, v := range knobs[k] {
			add([][2]string{{k, v}})
		}
	}
	for _, c := range combos {
		var fl [][2]string
		for _, kv := range strings.Split(c, ",") {
			k, v, ok := strings.Cut(strings.TrimSpace(kv), "=")
			if !ok || k == "" {
				return nil, fmt.Errorf("bundle %q: want flag=value,flag=value", c)
			}
			fl = append(fl, [2]string{k, v})
		}
		add(fl)
	}
	return out, nil
}

// Order is the run order of the configs in a round: reversed on odd rounds so
// no config always follows the same neighbour (A B C, then C B A).
func Order(n, round int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	if round%2 == 1 {
		for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
			idx[i], idx[j] = idx[j], idx[i]
		}
	}
	return idx
}

// Ratios is a config's per-round ratio to the control, gated as every other
// paired row here is (hb.Gate: at least six rounds, IQR/median at most 0.10).
// A latency or memory ratio below 1 is better, a rate above 1 is better.
type Ratios struct {
	Config                      string
	TTFTp50, TTFTp99, ITLp99    hb.RatioGate
	Goodput, AggRate, Mem       hb.RatioGate
	LoadSec                     hb.RatioGate
	MedianTTFT99, MedianITL99   float64
	MedianGoodput, MedianMemMiB float64
}

func ratio(a, b float64) float64 {
	if b == 0 {
		return math.NaN()
	}
	return a / b
}

// Compare reduces samples to each config's ratios against control, per round.
func Compare(samples []Sample, control string) []Ratios {
	byRound := map[string]map[int]Metrics{}
	var names []string
	for _, s := range samples {
		if byRound[s.Config] == nil {
			byRound[s.Config] = map[int]Metrics{}
			names = append(names, s.Config)
		}
		byRound[s.Config][s.Round] = s.M
	}
	var rounds []int
	for r := range byRound[control] {
		rounds = append(rounds, r)
	}
	sort.Ints(rounds)
	var out []Ratios
	for _, n := range names {
		var t50, t99, i99, gp, ag, mem, ld []float64
		var vt, vi, vg, vm []float64
		for _, r := range rounds {
			m, ok := byRound[n][r]
			c := byRound[control][r]
			if !ok {
				t50, t99, i99, gp, ag, mem, ld = append(t50, math.NaN()), append(t99, math.NaN()), append(i99, math.NaN()), append(gp, math.NaN()), append(ag, math.NaN()), append(mem, math.NaN()), append(ld, math.NaN())
				continue
			}
			t50 = append(t50, ratio(m.TTFTp50, c.TTFTp50))
			t99 = append(t99, ratio(m.TTFTp99, c.TTFTp99))
			i99 = append(i99, ratio(m.ITLp99, c.ITLp99))
			gp = append(gp, ratio(m.Goodput, c.Goodput))
			ag = append(ag, ratio(m.AggRate, c.AggRate))
			mem = append(mem, ratio(m.Mem(), c.Mem()))
			ld = append(ld, ratio(m.LoadSec, c.LoadSec))
			vt, vi, vg, vm = append(vt, m.TTFTp99), append(vi, m.ITLp99), append(vg, m.Goodput), append(vm, m.Mem())
		}
		out = append(out, Ratios{
			Config: n, TTFTp50: hb.Gate(t50), TTFTp99: hb.Gate(t99), ITLp99: hb.Gate(i99),
			Goodput: hb.Gate(gp), AggRate: hb.Gate(ag), Mem: hb.Gate(mem), LoadSec: hb.Gate(ld),
			MedianTTFT99: hb.Median(vt), MedianITL99: hb.Median(vi),
			MedianGoodput: hb.Median(vg), MedianMemMiB: hb.Median(vm),
		})
	}
	return out
}

// Dominates reports whether a is at least as good as b on every objective
// (p99 TTFT, p99 inter-token gap, memory lower; goodput higher) and strictly
// better on one.
func Dominates(a, b Ratios) bool {
	ge := a.MedianTTFT99 <= b.MedianTTFT99 && a.MedianITL99 <= b.MedianITL99 &&
		a.MedianMemMiB <= b.MedianMemMiB && a.MedianGoodput >= b.MedianGoodput
	gt := a.MedianTTFT99 < b.MedianTTFT99 || a.MedianITL99 < b.MedianITL99 ||
		a.MedianMemMiB < b.MedianMemMiB || a.MedianGoodput > b.MedianGoodput
	return ge && gt
}

// Pareto returns the configs no other config dominates, by medians over rounds.
func Pareto(rs []Ratios) map[string]bool {
	front := map[string]bool{}
	for i, a := range rs {
		dominated := false
		for j, b := range rs {
			if i != j && Dominates(b, a) {
				dominated = true
				break
			}
		}
		if !dominated {
			front[a.Config] = true
		}
	}
	return front
}

// Beats says a gated ratio is a real improvement: quotable, on the right side
// of 1 by more than its IQR. lowerBetter is true for latencies and memory.
func Beats(g hb.RatioGate, lowerBetter bool) bool {
	if g.Refused != "" {
		return false
	}
	m, q := float64(g.Median), float64(g.IQR)
	if lowerBetter {
		return m+q < 1
	}
	return m-q > 1
}

// Regresses is the mirror: a quotable ratio on the wrong side of 1 by more
// than its IQR.
func Regresses(g hb.RatioGate, lowerBetter bool) bool {
	if g.Refused != "" {
		return false
	}
	m, q := float64(g.Median), float64(g.IQR)
	if lowerBetter {
		return m-q > 1
	}
	return m+q < 1
}

// Verdict is the only sentence the tool says about a default: a config is a
// candidate to change it only when it has a paired, gated win on goodput or
// p99 TTFT and no gated regression of p99 TTFT, p99 gap, goodput or memory.
// Everything else, including a win that did not pass the gate, is not.
func Verdict(r Ratios) string {
	win := Beats(r.Goodput, false) || Beats(r.TTFTp99, true)
	loss := Regresses(r.TTFTp99, true) || Regresses(r.ITLp99, true) ||
		Regresses(r.Goodput, false) || Regresses(r.Mem, true)
	switch {
	case win && !loss:
		return "candidate (rerun to confirm before any default changes)"
	case loss:
		return "regresses"
	}
	for _, g := range []hb.RatioGate{r.TTFTp99, r.ITLp99, r.Goodput} {
		if g.Refused != "" {
			return "unresolved: " + g.Refused
		}
	}
	return "no resolvable difference"
}
