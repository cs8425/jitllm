package model

import (
	"fmt"

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
	// Path is how the history travelled: "host" when it came home to this
	// State's cache and went up again.
	Path string
	// Head is set when the output projection followed the blocks.
	Head bool
}

// MoveBlocks moves blocks [lo, hi) from the devices they are on to the device
// named to, while the sequence keeps its place: each block's history (its KV
// pages, a linear block's recurrent summary) goes with it, so the next token
// reads exactly the rows it would have read where it was, and no row of the
// prompt is computed again. It is the device-to-device relocation of the
// Scope's "move layers and pages between them", reachable by a caller rather
// than only through SetDevice, which brings everything home first.
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
	var mv BlockMove
	nd, ok := s.ld.(nn.NamedDevice)
	if !ok {
		return mv, fmt.Errorf("model: no device that places blocks by name is attached, " +
			"so there is no device to move blocks to")
	}
	if lo < s.lo || hi > s.hi || lo >= hi {
		return mv, fmt.Errorf("model: blocks %d..%d are outside this State's %d..%d", lo, hi-1, s.lo, s.hi-1)
	}
	for li := lo; li < hi; li++ {
		if !s.devAt(li) {
			return mv, fmt.Errorf("model: block %d is on the host; a move between devices "+
				"moves placed blocks, and SetGPULayers places one", li)
		}
		if glo, ghi := s.kvSpan(li); glo < lo || ghi > hi {
			return mv, fmt.Errorf("model: block %d shares its history with blocks %d..%d, "+
				"which move together", li, glo, ghi-1)
		}
		if s.pinnedOnDev(li) {
			return mv, fmt.Errorf("model: block %d is pinned where it is by the placement", li)
		}
	}
	headWas, headOn := nd.HeadDeviceName()
	headOn = headOn && s.head != nil
	mv.Path = "host"
	for li := lo; li < hi; li++ {
		if at, _ := nd.DeviceOf(li); at == nn.DeviceName(to) {
			continue
		}
		moved, err := s.moveBlockViaHost(nd, li, to)
		if err != nil {
			return mv, err
		}
		if !moved {
			mv.Declined = append(mv.Declined, li)
			continue
		}
		mv.Moved = append(mv.Moved, li)
		mv.KVBytes += s.blockHistoryAt(li, s.pos)
		mv.RecBytes += s.recBytesOf(li)
	}
	// The projection follows once nothing of this State is left beside it: a
	// head alone on a card costs a crossing a token and holds the card.
	if headOn && headWas != nn.DeviceName(to) && len(mv.Moved) > 0 && !s.anyBlockOn(nd, headWas) {
		took, err := nd.PrepHeadOn(to, s.newHead())
		if err != nil {
			return mv, err
		}
		mv.Head = took
	}
	return mv, nil
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
	per := uint64(pg.pp) * 4 / uint64(pg.p)
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
