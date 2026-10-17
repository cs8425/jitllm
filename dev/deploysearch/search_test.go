package main

import (
	"strings"
	"testing"
)

func TestGridIsOneFactorAtATimePlusBundles(t *testing.T) {
	g, err := Grid("bin", []string{"a", "b"}, map[string][]string{"a": {"1", "2"}, "b": {"x"}},
		[]string{"a=1,b=x", "a=1"})
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, c := range g {
		names = append(names, c.Name)
	}
	// "a=1" repeated as a bundle is deduplicated; the bundle is a candidate of its own.
	if got, want := strings.Join(names, " "), "default a=1 a=2 b=x a=1+b=x"; got != want {
		t.Fatalf("grid %q, want %q", got, want)
	}
	if args := g[4].Args(); strings.Join(args, " ") != "-a 1 -b x" {
		t.Fatalf("args %v", args)
	}
	if g[0].Args() != nil {
		t.Fatal("the control must add no flags")
	}
	if _, err := Grid("bin", nil, nil, []string{"novalue"}); err == nil {
		t.Fatal("a bundle without a value was accepted")
	}
}

func TestOrderReversesOnOddRounds(t *testing.T) {
	if got := Order(3, 0); got[0] != 0 || got[2] != 2 {
		t.Fatalf("round 0 %v", got)
	}
	if got := Order(3, 1); got[0] != 2 || got[2] != 0 {
		t.Fatalf("round 1 %v", got)
	}
}

func rounds(cfg string, n int, m Metrics) []Sample {
	var s []Sample
	for r := 1; r <= n; r++ {
		s = append(s, Sample{Config: cfg, Round: r, M: m})
	}
	return s
}

func TestVerdictNeedsAGatedWinAndNoRegression(t *testing.T) {
	ctl := Metrics{TTFTp99: 1000, ITLp99: 50, Goodput: 4, HostMiB: 1000}
	var ss []Sample
	ss = append(ss, rounds("default", 6, ctl)...)
	// Better goodput and TTFT, nothing worse.
	win := ctl
	win.Goodput, win.TTFTp99 = 5, 700
	ss = append(ss, rounds("win", 6, win)...)
	// Better TTFT but much more memory.
	mem := win
	mem.HostMiB = 2000
	ss = append(ss, rounds("mem", 6, mem)...)
	// The same as the control.
	ss = append(ss, rounds("aa", 6, ctl)...)
	// A win seen in only three rounds is refused, not quoted.
	ss = append(ss, rounds("short", 3, win)...)

	got := map[string]string{}
	rs := Compare(ss, "default")
	for _, r := range rs {
		got[r.Config] = Verdict(r)
	}
	if !strings.HasPrefix(got["win"], "candidate") {
		t.Errorf("win: %q", got["win"])
	}
	if got["mem"] != "regresses" {
		t.Errorf("mem: %q", got["mem"])
	}
	if got["aa"] != "no resolvable difference" {
		t.Errorf("aa: %q", got["aa"])
	}
	if !strings.HasPrefix(got["short"], "unresolved") {
		t.Errorf("short: %q", got["short"])
	}
}

func TestParetoDropsDominatedConfigs(t *testing.T) {
	ctl := Metrics{TTFTp99: 1000, ITLp99: 50, Goodput: 4, HostMiB: 1000}
	fast := ctl
	fast.TTFTp99 = 500
	lean := ctl
	lean.HostMiB = 500
	worse := ctl
	worse.TTFTp99, worse.Goodput = 1500, 3
	var ss []Sample
	for name, m := range map[string]Metrics{"default": ctl, "fast": fast, "lean": lean, "worse": worse} {
		ss = append(ss, rounds(name, 6, m)...)
	}
	front := Pareto(Compare(ss, "default"))
	// fast and lean trade latency for memory, so both stay; the control is
	// dominated by each, and worse by the control.
	if !front["fast"] || !front["lean"] {
		t.Fatalf("front %v", front)
	}
	if front["default"] || front["worse"] {
		t.Fatalf("a dominated config is on the front: %v", front)
	}
}
