//go:build darwin

package sched

import (
	"runtime"
	"sync"

	"github.com/jitllm/jitllm/jit/gpu/ffi"
)

// qosBackground is QOS_CLASS_BACKGROUND (sys/qos.h). macOS exposes no
// processor affinity; on Apple Silicon its scheduler keeps a background-QoS
// thread on the efficiency cluster, which is the only way to put work there.
const qosBackground = 0x09

var (
	qosOnce sync.Once
	qosSet  func(uint32, int32) int32
)

// EQoS puts the calling goroutine's thread at background QoS and locks the
// goroutine to that thread, so what it runs stays on the E-cores. It reports
// whether the class was set. The lock is for the goroutine's life: a thread
// whose class was lowered must not go back to the runtime's pool, where it
// would carry the class to whatever ran on it next (the leak RULE 5 records
// for an affinity mask). A pool's workers call it once (New, on a pool wholly
// on the E-cores); a caller that drives such a pool calls it on the goroutine
// it drives the pool from, which is participant 0 of every region.
func EQoS() bool {
	qosOnce.Do(func() {
		l, err := ffi.Open("/usr/lib/libSystem.B.dylib")
		if err != nil || !l.Has("pthread_set_qos_class_self_np") {
			return
		}
		qosSet = ffi.Fn2[int32, uint32, int32](l, "pthread_set_qos_class_self_np")
	})
	if qosSet == nil {
		return false
	}
	runtime.LockOSThread()
	return qosSet(qosBackground, 0) == 0
}

// allE reports whether every cpu of a pool is an E-core index (ECores).
func allE(cpus []int) bool {
	e := ECores()
	if len(e) == 0 || len(cpus) == 0 {
		return false
	}
	for _, c := range cpus {
		if c < e[0] {
			return false
		}
	}
	return true
}
