package main

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

// Side is one engine of a race: where it serves, what it is called on
// screen, and the request it is sent.
type Side struct {
	Name  string // the engine, as the pane and the summary name it
	Label string // a second line under the name: version, backend, format
	Req   hb.Request
}

// Event is what a race reports while it runs, one side at a time.
type Event struct {
	Side int
	At   time.Duration // since the start of the race
	Text string        // a text chunk; empty on the final event
	Done *hb.Sample    // set once, when the side's stream has ended
}

// Race sends both sides' requests at the same instant and reports every text
// chunk and each side's final sample on events, which it closes when both
// have finished. Each side has its own client, so neither waits on the
// other's connection.
//
// The start is a barrier: both goroutines have built their request and are
// parked on one channel, and closing it releases them together, so the gap
// between the two sends is the scheduler's, not the setup's.
func Race(ctx context.Context, sides [2]Side, clients [2]*http.Client, events chan<- Event) [2]hb.Sample {
	var out [2]hb.Sample
	var wg sync.WaitGroup
	gate := make(chan struct{})
	var t0 time.Time
	for i := range sides {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := sides[i].Req
			r.OnText = func(text string) {
				if events != nil {
					events <- Event{Side: i, At: time.Since(t0), Text: text}
				}
			}
			<-gate
			s := hb.Do(ctx, clients[i], r)
			s.Arm = sides[i].Name
			out[i] = s
			if events != nil {
				events <- Event{Side: i, At: time.Since(t0), Done: &out[i]}
			}
		}()
	}
	t0 = time.Now()
	close(gate)
	wg.Wait()
	if events != nil {
		close(events)
	}
	return out
}

// Warm sends n throwaway requests to both sides at once, as the race will,
// so neither engine is timed on a first use (a JIT tune, a graph capture, a
// cold connection). It returns the first error.
func Warm(ctx context.Context, sides [2]Side, clients [2]*http.Client, n int) error {
	for range n {
		for i, s := range Race(ctx, sides, clients, nil) {
			if s.Err != "" {
				return fmt.Errorf("%s: warm-up: %s", sides[i].Name, s.Err)
			}
		}
	}
	return nil
}

// liveRate is a running side's decode rate: chunks after the first over the
// time since the first. A chunk is a token on every engine this was run
// against; the final rate is the usage count's (hb.Sample.DecodeRate).
func liveRate(chunks int, first, now time.Duration) float64 {
	if chunks < 2 || now <= first {
		return 0
	}
	return float64(chunks-1) / (now - first).Seconds()
}

// Summary is the one line a recording ends on: each side's TTFT and decode
// rate, then the left side over the right on both.
func Summary(sides [2]Side, s [2]hb.Sample) string {
	var b strings.Builder
	for i := range s {
		if i > 0 {
			b.WriteString(" · ")
		}
		if s[i].Err != "" {
			fmt.Fprintf(&b, "%s: error", sides[i].Name)
			continue
		}
		fmt.Fprintf(&b, "%s: TTFT %s, %.1f tok/s", sides[i].Name, msOf(s[i].TTFT), s[i].DecodeRate())
	}
	if s[0].Err == "" && s[1].Err == "" {
		r := s[0].DecodeRate() / s[1].DecodeRate()
		t := float64(s[0].TTFT) / float64(s[1].TTFT)
		if !math.IsNaN(r) && !math.IsInf(r, 0) {
			fmt.Fprintf(&b, " · %s/%s: decode %.2fx, TTFT %.2fx", sides[0].Name, sides[1].Name, r, t)
		}
	}
	return b.String()
}

func msOf(d time.Duration) string {
	return fmt.Sprintf("%.0f ms", float64(d)/float64(time.Millisecond))
}
