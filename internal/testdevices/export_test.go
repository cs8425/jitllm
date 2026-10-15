package testdevices

import "testing"

// SetForTest puts spec in force as the selection until the test ends, as if
// JITLLM_TEST_DEVICES had been set to it, with a clean ledger. An empty spec
// is no selection.
func SetForTest(t *testing.T, spec string) {
	t.Helper()
	was, wasSel := selection, selected
	selection, selected = nil, false
	if spec != "" {
		ds, err := parse(spec)
		if err != nil {
			t.Fatal(err)
		}
		selection, selected = ds, true
	}
	Reset()
	t.Cleanup(func() {
		selection, selected = was, wasSel
		Reset()
	})
}
