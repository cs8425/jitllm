package model

import (
	"errors"
	"fmt"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
)

// BlockMove is what MoveBlocks did.
type BlockMove struct {
	// Moved is the blocks that changed device; Declined the ones the target
	// refused, which stay where they were with their history.
	Moved    []int
	Declined []int
	// KVBytes is the history the moved blocks carried, at the position the
	// move was made.
	KVBytes uint64
	// RecBytes is the recurrent summaries the moved linear blocks carried.
	RecBytes uint64
	// Path is how the history travelled, the slowest any block took: "peer"
	// card to card, "staged" through a host buffer the tier holds, "host"
	// home to this State's cache and up again; "" when there was none.
	Path string
	// Copied is the bytes the tier copied between the devices (the history
	// in the devices' own page layout).
	Copied uint64
	// Head is set when the output projection followed the blocks.
	Head bool
	// Staged is the blocks prepared on the target while the sequence kept
	// generating (PrepareMove), and Prepare how long that took: time the
	// sequence did not wait.
	Staged  int
	Prepare time.Duration
	// Pause is the commit: what the sequence waited between two tokens.
	Pause time.Duration
}

// PendingMove is a move between devices being prepared while the State keeps
// generating: the target takes each block's weights and builds its kernels
// on another goroutine (nn.BlockMover.StageBlock), and Commit, at a token
// boundary, copies the history and switches the route. Ready says the
// preparation is done, so a caller that commits only then never waits on it.
type PendingMove struct {
	s      *State
	lo, hi int
	to     string
	// blocks are the ones to move, plans their plans, staged which the
	// target prepared ahead.
	blocks []int
	plans  []nn.LayerPlan
	staged []bool
	done   chan struct{}
	prep   time.Duration
	closed bool
	// headStaged is the projection prepared on the target as well.
	headStaged bool
}

// MoveBlocks moves blocks [lo, hi) from the devices they are on to the device
// named to, while the sequence keeps its place: each block's history (its KV
// pages, a linear block's recurrent summary) goes with it, so the next token
// reads exactly the rows it would have read where it was, and no row of the
// prompt is computed again. It is the device-to-device relocation of the
// Scope's "move layers and pages between them", reachable by a caller rather
// than only through SetDevice, which brings everything home first.
//
// It is PrepareMove and Commit at once: the target is prepared inside the
// call. A caller that can keep generating meanwhile prepares first and
// commits at a later token.
//
// Call it between tokens: like every placement call it must not overlap a
// Forward of this State. Blocks already on to are left alone, and a block the
// target refuses stays where it was and is reported. When no block of this
// State is left on the device holding the output projection, the projection
// follows them.
//
// Every block of the range must be on a device: a host block is placed by
// SetGPULayers (or WithPlacement) instead, and a range that does not hold a
// KV-sharing group whole is refused, since the group's readers attend to its
// source's history wherever it is.
func (s *State) MoveBlocks(lo, hi int, to string) (BlockMove, error) {
	p, err := s.PrepareMove(lo, hi, to)
	if err != nil {
		return BlockMove{}, err
	}
	return p.Commit()
}

// PrepareMove starts a move of blocks [lo, hi) to the device named to (see
// MoveBlocks) and returns at once: the target prepares the blocks while the
// State keeps generating on the source, and nothing the State runs changes
// until Commit. Call it, Commit and Cancel between tokens; a State holds one
// pending move at a time.
func (s *State) PrepareMove(lo, hi int, to string) (*PendingMove, error) {
	nd, ok := s.ld.(nn.NamedDevice)
	if !ok {
		return nil, fmt.Errorf("model: no device that places blocks by name is attached, " +
			"so there is no device to move blocks to")
	}
	if s.pending != nil {
		return nil, fmt.Errorf("model: a move of blocks %d..%d to %s is pending",
			s.pending.lo, s.pending.hi-1, s.pending.to)
	}
	if lo < s.lo || hi > s.hi || lo >= hi {
		return nil, fmt.Errorf("model: blocks %d..%d are outside this State's %d..%d", lo, hi-1, s.lo, s.hi-1)
	}
	for li := lo; li < hi; li++ {
		if !s.devAt(li) {
			return nil, fmt.Errorf("model: block %d is on the host; a move between devices "+
				"moves placed blocks, and SetGPULayers places one", li)
		}
		if glo, ghi := s.kvSpan(li); glo < lo || ghi > hi {
			return nil, fmt.Errorf("model: block %d shares its history with blocks %d..%d, "+
				"which move together", li, glo, ghi-1)
		}
		if s.pinnedOnDev(li) {
			return nil, fmt.Errorf("model: block %d is pinned where it is by the placement", li)
		}
	}
	p := &PendingMove{s: s, lo: lo, hi: hi, to: to, done: make(chan struct{})}
	for li := lo; li < hi; li++ {
		if at, _ := nd.DeviceOf(li); at == nn.DeviceName(to) {
			continue
		}
		p.blocks = append(p.blocks, li)
		p.plans = append(p.plans, *s.planFor(li))
	}
	p.staged = make([]bool, len(p.blocks))
	s.pending = p
	mover, ok := s.ld.(nn.BlockMover)
	// Staging reads the block's page and hands the tier its weights while the
	// State runs: the State's own vision and encoder segments, and a mixture's
	// expert placement, are decided by the offer path, so they are prepared at
	// the commit instead.
	if !ok || s.m.opt.hostMoves || s.vis != nil || s.enc != nil || s.c.MoE() || len(p.blocks) == 0 {
		close(p.done)
		return p, nil
	}
	// The projection is prepared too when the move leaves none of the State's
	// blocks beside it (Commit then moves it).
	var head *nn.Head
	if was, on := nd.HeadDeviceName(); on && s.head != nil && was != nn.DeviceName(to) {
		follows := true
		for li := s.lo; li < s.hi; li++ {
			if at, ok := nd.DeviceOf(li); ok && s.devAt(li) && at == was && (li < lo || li >= hi) {
				follows = false
			}
		}
		if follows {
			head = s.newHead()
		}
	}
	// Room for the history at the commit: the tokens the preparation runs
	// beside add to it.
	pos := s.pos + 256
	go func() {
		defer close(p.done)
		t0 := time.Now()
		all := true
		for i, li := range p.blocks {
			w, err := s.blockWeights(li)
			if err != nil {
				all = false
				break
			}
			p.staged[i] = mover.StageBlock(li, to, &p.plans[i], &w, pos)
			all = all && p.staged[i]
		}
		if all && head != nil {
			p.headStaged = mover.StageHead(to, head)
		}
		p.prep = time.Since(t0)
	}()
	return p, nil
}

// Ready reports that the target has finished preparing, so Commit will not
// wait for it.
func (p *PendingMove) Ready() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

// Cancel abandons a move that was not committed, freeing what the target
// prepared for it.
func (p *PendingMove) Cancel() {
	if p.closed {
		return
	}
	<-p.done
	p.closed = true
	p.s.pending = nil
	if mover, ok := p.s.ld.(nn.BlockMover); ok {
		mover.Unstage(p.lo, p.hi)
	}
}

// Commit makes the move, between two tokens: each block's history goes to
// the target and the route switches. It waits for the preparation if it is
// still running.
func (p *PendingMove) Commit() (mv BlockMove, err error) {
	s := p.s
	if p.closed {
		return mv, fmt.Errorf("model: the move was already committed or cancelled")
	}
	<-p.done
	p.closed = true
	s.pending = nil
	t0 := time.Now()
	mv.Prepare = p.prep
	for _, st := range p.staged {
		if st {
			mv.Staged++
		}
	}
	defer func() { mv.Pause = time.Since(t0) }()
	nd := s.ld.(nn.NamedDevice)
	headWas, headOn := nd.HeadDeviceName()
	headOn = headOn && s.head != nil
	mover, direct := s.ld.(nn.BlockMover)
	direct = direct && !s.m.opt.hostMoves && s.nseq <= 1
	for i, li := range p.blocks {
		if at, _ := nd.DeviceOf(li); at == nn.DeviceName(p.to) {
			continue
		}
		path, copied, moved := "", uint64(0), false
		err = nn.ErrNotDirect
		if direct {
			path, copied, moved, err = s.moveBlockDirect(mover, li, p.to, &p.plans[i])
		}
		if errors.Is(err, nn.ErrNotDirect) {
			path = "host"
			moved, err = s.moveBlockViaHost(nd, li, p.to)
		}
		if err != nil {
			if mover != nil {
				mover.Unstage(p.lo, p.hi)
			}
			return mv, err
		}
		if !moved {
			mv.Declined = append(mv.Declined, li)
			continue
		}
		mv.Moved = append(mv.Moved, li)
		mv.KVBytes += s.blockHistoryAt(li, s.pos)
		mv.RecBytes += s.recBytesOf(li)
		mv.Copied += copied
		mv.Path = slowerPath(mv.Path, path)
	}
	if mover != nil {
		// A declined block's staged copy, if the target kept one, and what
		// the moved blocks left on their sources.
		mover.Unstage(p.lo, p.hi)
		defer mover.RetireMoved()
	}
	// The projection follows once nothing of this State is left beside it: a
	// head alone on a card costs a crossing a token and holds the card.
	if headOn && headWas != nn.DeviceName(p.to) && len(mv.Moved) > 0 && !s.anyBlockOn(nd, headWas) {
		if mv.Head, err = nd.PrepHeadOn(p.to, s.newHead()); err != nil {
			return mv, err
		}
	}
	return mv, nil
}

// slowerPath is the slower of two history paths, "" being none.
func slowerPath(a, b string) string {
	rank := map[string]int{"": 3, "peer": 2, "staged": 1, "host": 0}
	if rank[b] < rank[a] {
		return b
	}
	return a
}

// moveBlockDirect moves block li through the tier: the history is copied from
// the source's pages to the target's and never comes home. A linear block's
// summary, a few kilobytes, comes home and goes up as it does on every other
// move. nn.ErrNotDirect changed nothing.
func (s *State) moveBlockDirect(mover nn.BlockMover, li int, to string, plan *nn.LayerPlan) (string, uint64, bool, error) {
	if !s.migrateRec(li, false) {
		return "", 0, false, fmt.Errorf("model: block %d's recurrent summary would not come home", li)
	}
	res, err := mover.MoveBlock(li, to, plan, nil, s.pos)
	if errors.Is(err, nn.ErrNotStaged) {
		var w nn.LayerWeights
		if w, err = s.blockWeights(li); err != nil {
			return "", 0, false, err
		}
		res, err = mover.MoveBlock(li, to, plan, &w, s.pos)
	}
	if err != nil || !res.Moved {
		return "", 0, false, err
	}
	if !s.migrateRec(li, true) {
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		return "", 0, false, fmt.Errorf("model: block %d reached %s but its recurrent summary did not; "+
			"it runs on the host", li, to)
	}
	s.releasePlacedPage(s.ld, li)
	return res.Path, res.Bytes, true, nil
}

// moveBlockViaHost moves block li to the device named to through this State's
// own cache: the history comes home (MigrateKV copies, so the source keeps
// its rows until the target has them), the target takes the block, which
// frees it on the source, and the history goes up to the target. False means
// the target refused the block, which then stays where it was.
func (s *State) moveBlockViaHost(nd nn.NamedDevice, li int, to string) (bool, error) {
	if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
		return false, fmt.Errorf("model: block %d's history would not come home from its device", li)
	}
	plan, w, err := s.blockOffer(li)
	if err != nil {
		return false, err
	}
	took, err := nd.PrepLayerOn(to, li, &plan, &w)
	if err != nil || !took {
		return false, err
	}
	if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
		// The weights are on the target with no history; the history is home,
		// so the block runs there instead of attending over empty pages.
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		return false, fmt.Errorf("model: block %d reached %s but its history did not; "+
			"it runs on the host", li, to)
	}
	s.releasePlacedPage(s.ld, li)
	return true, nil
}

// anyBlockOn reports whether a block of this State is on the named device.
func (s *State) anyBlockOn(nd nn.NamedDevice, name string) bool {
	for li := s.lo; li < s.hi; li++ {
		if !s.devAt(li) {
			continue
		}
		if at, ok := nd.DeviceOf(li); ok && at == name {
			return true
		}
	}
	return false
}

// DeviceOf is the name of the device holding block li, and whether a device
// holds it.
func (s *State) DeviceOf(li int) (string, bool) {
	if !s.devAt(li) {
		return "", false
	}
	if nd, ok := s.ld.(nn.NamedDevice); ok {
		return nd.DeviceOf(li)
	}
	return "", true
}

// blockHistoryAt is historyAt for block li alone.
func (s *State) blockHistoryAt(li, pos int) uint64 {
	if li < s.kv.lo || li >= len(s.kv.layers) {
		return 0
	}
	pg := &s.kv.layers[li]
	if pg.p == 0 {
		return 0
	}
	n := pos
	if pg.win > 0 && n > pg.win+pg.p {
		n = pg.win + pg.p
	}
	per := uint64(pg.pp) * 4 / uint64(pg.p) // a key's and a value's row
	if pg.latent {
		return uint64(n) * per
	}
	return 2 * uint64(n) * per
}

// recBytesOf is block li's recurrent summary in bytes, every row's.
func (s *State) recBytesOf(li int) uint64 {
	if s.rconv == nil || li >= len(s.rconv) {
		return 0
	}
	return uint64(len(s.rconv[li])+len(s.rstate[li])) * 4
}
