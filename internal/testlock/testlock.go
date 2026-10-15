// Package testlock serialises GPU tests across processes.
//
// `go test ./...` runs packages in parallel, and packages that open a device
// contend for VRAM and the driver, failing in a different package each run. A
// file lock costs nothing when uncontended and turns the failure into a wait.
//
// Tests only: nothing in the engine takes it.
package testlock

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
)

// GPU takes the cross-process GPU test lock and releases it when t ends. Every
// package that allocates on a card takes the same file, so two of them do not
// allocate on one card at the same time.
//
// With JITLLM_TEST_DEVICES set the test binary touches only the devices it
// names (internal/testdevices), so the lock is one file per selected device,
// taken in a fixed order: two gates on different cards of one box do not wait
// for each other, and two on the same card still do. With it unset, one file
// serves the whole box.
//
// A lock that cannot be taken is no lock rather than no test: the test runs.
func GPU(t testing.TB) {
	t.Helper()
	names := []string{"jitllm-gpu-test.lock"}
	if testdevices.Selected() {
		names = names[:0]
		sel := testdevices.Selection()
		sort.Strings(sel)
		for _, d := range sel {
			names = append(names, "jitllm-gpu-test-"+strings.ReplaceAll(d, ":", "")+".lock")
		}
	}
	for _, n := range names {
		f, err := os.OpenFile(filepath.Join(os.TempDir(), n), os.O_CREATE|os.O_RDWR, 0o666)
		if err != nil {
			continue
		}
		if err := lock(f); err != nil {
			f.Close()
			continue
		}
		t.Cleanup(func() {
			unlock(f)
			f.Close()
		})
	}
}
