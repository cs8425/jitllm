package server

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// The orchestrator (docs/design/orchestrator.md): the server moving blocks
// itself. When a session starts a generate, its host blocks are offered to
// its model's devices (promote); when a card declines for lack of room, the
// least deserving idle sessions on the cards -- other models' first, lowest
// priority, then idle longest -- bring their blocks home with their history
// (demote), and the offer is made again. A running session is never a victim,
// a session or model the caller placed is never moved, and nothing is
// refused: a promotion that does not fit runs where it is.

// orchLogSize bounds the move log the stats carry.
const orchLogSize = 256

// OrchMove is one orchestrated move.
type OrchMove struct {
	At      time.Time
	Session string
	Model   string
	// Blocks moved, To "device" or "host", and Why in words.
	Blocks int
	To     string
	Why    string
	Millis float64
}

// orchestrator is the Engine's record of what it moved.
type orchestrator struct {
	mu   sync.Mutex
	log  []OrchMove
	next int
	// Counters: promotions and demotions made, and blocks they moved.
	promotions, demotions atomic.Int64
	blocksUp, blocksHome  atomic.Int64
}

func (o *orchestrator) note(m OrchMove) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.log) < orchLogSize {
		o.log = append(o.log, m)
		return
	}
	o.log[o.next] = m
	o.next = (o.next + 1) % orchLogSize
}

// OrchMoves is the orchestrator's recent moves, oldest first.
func (e *Engine) OrchMoves() []OrchMove {
	o := &e.orch
	o.mu.Lock()
	defer o.mu.Unlock()
	out := make([]OrchMove, 0, len(o.log))
	out = append(out, o.log[o.next:]...)
	return append(out, o.log[:o.next]...)
}

// OrchCounts is what the orchestrator has done: promotions, demotions, and
// the blocks they moved up and home.
func (e *Engine) OrchCounts() (promotions, demotions, blocksUp, blocksHome int64) {
	o := &e.orch
	return o.promotions.Load(), o.demotions.Load(), o.blocksUp.Load(), o.blocksHome.Load()
}

// Unpin hands a session's placement back to the orchestrator.
func (s *Session) Unpin() { s.pinned.Store(false) }

// promote offers s's host blocks to its model's devices, demoting idle
// sessions to make room when the cards decline. The caller holds s.mu.
func (e *Engine) promote(s *Session) error {
	lm := s.lm
	if e.cfg.NoOrchestrate || s.pinned.Load() || lm.callerPlaced || lm.device() == nil {
		return nil
	}
	n := lm.m.Cfg.NLayer
	if s.st.GPULayers() >= n {
		return nil
	}
	if err := e.bindSessionLocked(s); err != nil {
		return nil // a session that cannot attach runs where it is
	}
	place := func(why string) int {
		start := time.Now()
		before := s.st.GPULayers()
		s.st.PlaceBlocks(0, n)
		got := s.st.GPULayers() - before
		if got > 0 {
			e.orch.promotions.Add(1)
			e.orch.blocksUp.Add(int64(got))
			e.orch.note(OrchMove{At: start, Session: s.id, Model: lm.id, Blocks: got, To: "device", Why: why,
				Millis: float64(time.Since(start).Microseconds()) / 1000})
		}
		return got
	}
	place("promote: generating")
	for s.st.GPULayers() < n {
		v := e.demotionVictim(s)
		if v == nil {
			break
		}
		if !e.demote(v, "make room for "+s.id) {
			continue
		}
		if place("promote: room made") == 0 {
			// Room taken from that victim did not let a block in: the card's
			// shortfall is elsewhere; stop rather than empty every card.
			break
		}
	}
	s.refresh()
	return nil
}

// demotionVictim is the least deserving session holding device blocks that
// is not s, not running, not pinned and not caller-placed: other models'
// sessions first, then the lowest priority, then the one idle longest. nil
// when there is none.
func (e *Engine) demotionVictim(s *Session) *Session {
	e.mu.RLock()
	var cand []*Session
	for _, v := range e.sessions {
		if v == s || v.closed.Load() || v.running.Load() || v.pinned.Load() || v.lm.callerPlaced ||
			v.snapDevBlocks.Load() == 0 {
			continue
		}
		cand = append(cand, v)
	}
	e.mu.RUnlock()
	sort.Slice(cand, func(i, j int) bool {
		a, b := cand[i], cand[j]
		if oa, ob := a.lm != s.lm, b.lm != s.lm; oa != ob {
			return oa
		}
		if pa, pb := a.priority.Load(), b.priority.Load(); pa != pb {
			return pa < pb
		}
		return a.lastUsed.Load() < b.lastUsed.Load()
	})
	if len(cand) == 0 {
		return nil
	}
	return cand[0]
}

// demote brings v's blocks home with their history, when v's lock is free
// (an idle session; a busy one is skipped, never waited on). It reports
// whether it moved anything.
func (e *Engine) demote(v *Session, why string) bool {
	if !v.mu.TryLock() {
		return false
	}
	defer v.mu.Unlock()
	start := time.Now()
	n, _ := v.st.ReleaseBlocks(0, v.lm.m.Cfg.NLayer)
	v.refresh()
	if n == 0 {
		return false
	}
	e.orch.demotions.Add(1)
	e.orch.blocksHome.Add(int64(n))
	e.orch.note(OrchMove{At: start, Session: v.id, Model: v.lm.id, Blocks: n, To: "host", Why: why,
		Millis: float64(time.Since(start).Microseconds()) / 1000})
	return true
}
