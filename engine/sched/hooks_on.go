//go:build jitllmtest

package sched

import "sync/atomic"

// onWorkerWake, set by a test (-tags jitllmtest), runs in worker id after it
// has seen a region and before it tries to join it: a test that blocks there
// stands for a worker the OS or the Go scheduler keeps off a CPU. Atomic,
// because a worker let go reads it while the test that set it clears it.
var onWorkerWake atomic.Pointer[func(p *Pool, id int)]

func workerWake(p *Pool, id int) {
	if f := onWorkerWake.Load(); f != nil {
		(*f)(p, id)
	}
}

// OnPhase, set by a test (-tags jitllmtest), is told each fused phase's total
// and chunk on the caller's goroutine, so a test can name where its phases
// come from (runtime.Callers).
var OnPhase func(total, chunk int)

func phaseHook(total, chunk int) {
	if f := OnPhase; f != nil {
		f(total, chunk)
	}
}
