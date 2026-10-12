package model

import (
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/nn"
)

// A decode token runs each run of consecutive resident host blocks as ONE pool
// region (sched.Pool.Fuse): the workers join once and every op of every block
// in the run -- the norms, the projections, the rotary and cache write,
// attention, the activation, the residual adds -- is a phase of that region
// rather than a region of its own. The kernels, their chunks and their order
// are the per-op path's, so the answer is the same bits
// (TestFusedBlocksMatchPerOp); what goes is the round trip per op.
//
// The run is bridged across blocks for as long as nothing ends it: a device
// run, a block that has to be read from the container first, or the end of
// the token. A relocation, an eviction or a seam move needs no re-link,
// because the run is not compiled: it is the same per-block code, so the next
// token's runs follow the next token's placement and residency as they are.

// fuseRun opens a fused run unless one is open or this State runs per op.
func (s *State) fuseRun() {
	if s.fused || s.fuseOff {
		return
	}
	// A run starts with nothing to look ahead to: a set named in an earlier
	// run is the State's to refill now, and no worker may be reading it.
	s.jit.SetAhead(nil)
	s.jit.SetLookahead(!s.lookOff)
	if s.jit.Fuse() {
		s.fused = true
		s.FusedRuns++
		fusedRunsTotal.Add(1)
	}
}

// unfuse closes the open fused run, if there is one.
func (s *State) unfuse() {
	if s.fused {
		s.fused = false
		s.jit.Unfuse()
	}
}

// SetBlockFusion turns fused host runs on or off for this State's decode
// tokens (WithBlockFusion is the model-wide default): the A/B arm and the
// gates' per-op reference.
func (s *State) SetBlockFusion(on bool) { s.fuseOff = !on }

// blockResident reports whether block li's page is held, so a fused run does
// not stay open across the read of one that is not.
func (m *Model) blockResident(li int) bool {
	c := m.container
	return c == nil || li >= c.NPages() || c.Resident(li)
}

// fusedRunsTotal counts fused runs across every State, for the gates that must
// show a fused run happened inside them.
var fusedRunsTotal atomic.Int64

// groupOf runs n matvecs of ws that read x as one region
// (nn.MatVecPackedGroup) when a fused run is open, and reports whether it did;
// false means nothing ran and the caller runs them one at a time. Inside a
// fused run this folds the activation's quantization into the products and
// runs siblings (q/k/v, gate/up) back to back, with no phase wait between.
func (s *State) groupOf(x []float32, ws [nn.MaxGroup]*tensor, outs [nn.MaxGroup][]float32, n int) bool {
	if !s.fused || s.groupOff {
		return false
	}
	g := &s.grp
	for i := 0; i < n; i++ {
		w := ws[i]
		if w.packed == nil || w.k != ws[0].k || len(x) < w.k {
			return false
		}
		g.Out[i], g.T[i], g.P[i], g.Rows[i] = outs[i], w.typ, w.packed, w.rows
	}
	g.N, g.X, g.K = n, x, ws[0].k
	ok := s.jit.MatVecPackedGroup(g)
	g.X = nil
	for i := range g.Out {
		g.Out[i], g.P[i] = nil, nil
	}
	if ok {
		s.Grouped++
	}
	return ok
}

// aheadOps is how many lookahead sets a block has: what it reads after its
// q/k/v, after its o, after its gate/up and after its down.
const aheadOps = 4

// aheadOf names in the fused run's lookahead (nn.AheadSet) the weights block
// li reads after op: idle workers at the end of a phase hint them into the
// outer caches. It does nothing outside a fused run. The set is the State's,
// one per (block, op), named once per run (fuseRun clears the last one), so
// the set a worker may still be reading is never the one being filled.
func (s *State) aheadOf(li, op int, ws [nn.MaxGroup]*tensor) {
	if !s.fused || s.lookOff {
		return
	}
	i := li*aheadOps + op
	if i >= len(s.aheads) {
		if n := len(s.m.layers) * aheadOps; n > i {
			s.aheads = make([]nn.AheadSet, n)
		} else {
			s.jit.SetAhead(nil)
			return
		}
	}
	set := &s.aheads[i]
	for k, w := range ws {
		set.P[k] = nil
		if w != nil {
			set.P[k] = w.packed
		}
	}
	s.jit.SetAhead(set)
}

// aheadNext names block li+1's q/k/v as what follows li's down projection,
// when that block runs next in this fused run: on the host and resident.
func (s *State) aheadNext(li int) {
	n := li + 1
	if n >= s.hi || s.devAt(n) || !s.m.blockResident(n) || s.m.layers[n].wq.packed == nil {
		s.aheadOf(li, 3, [nn.MaxGroup]*tensor{})
		return
	}
	l := &s.m.layers[n]
	s.aheadOf(li, 3, [nn.MaxGroup]*tensor{&l.wq, &l.wk, &l.wv})
}

// SetLookahead turns the fused runs' lookahead on or off for this State's
// decode tokens (WithLookahead is the model-wide default).
func (s *State) SetLookahead(on bool) { s.lookOff = !on }

// SetMatVecGroups turns groupOf on or off inside fused runs: off, a fused
// run's matvecs and their activation quantization are a phase each.
func (s *State) SetMatVecGroups(on bool) { s.groupOff = !on }
