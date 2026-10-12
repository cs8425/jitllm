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

// SetMatVecGroups turns groupOf on or off inside fused runs: off, a fused
// run's matvecs and their activation quantization are a phase each.
func (s *State) SetMatVecGroups(on bool) { s.groupOff = !on }
