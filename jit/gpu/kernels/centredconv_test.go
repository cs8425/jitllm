package kernels

import "testing"

// TestConv3CentredValidates: the kernel builds and passes ir.Validate (its
// input and output are separate params), and refuses an empty shape. backend
// runs it against the host's loop (TestConv3CentredMatchesTheLoop).
func TestConv3CentredValidates(t *testing.T) {
	for _, tc := range []struct{ chans, rows int }{{8, 1}, {8, 6}, {1024, 64}, {3, 7}} {
		k, err := Conv3Centred(tc.chans, tc.rows)
		if err != nil {
			t.Fatal(err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("Conv3Centred(%d, %d): %v", tc.chans, tc.rows, err)
		}
	}
	if _, err := Conv3Centred(0, 4); err == nil {
		t.Error("an empty shape built")
	}
}
