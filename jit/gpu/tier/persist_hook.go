//go:build jitllmtest

package tier

// PersistByDefaultForTest makes every tier opened after it start with
// Config.Persist set, and returns the default it replaced: the regression
// gates (relocation, paging, batching, sessions, seam moves, the allocation
// gate) run through the persistent path without each naming it. Not safe
// against a concurrent open.
func PersistByDefaultForTest(on bool) bool {
	old := persistDefault
	persistDefault = on
	return old
}

// PersistLaunchesForTest is how many persistent kernels this process has
// launched, for a gate that must show the path ran.
func PersistLaunchesForTest() int64 { return persistTotal.Load() }
