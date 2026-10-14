package sched

import "testing"

// TestFuseWhileTheOtherViewIsParked: a session sending its prefill elsewhere
// parks its decode view (Park), which makes the other session's view its
// cores' only user for a fused region; Unpark takes the user back, and a view
// closed while parked leaves the counts as if it had never been one.
func TestFuseWhileTheOtherViewIsParked(t *testing.T) {
	a := Shared([]int{0, 1, 2})
	b := Shared([]int{0, 1, 2})
	defer a.Close()
	defer b.Close()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("Fuse opened beside a view that is not parked: this gate is not on the case it names")
	}
	b.Park()
	if !a.Fuse() {
		t.Fatal("Fuse refused beside a parked view")
	}
	a.Unfuse()
	b.Unpark()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("Fuse opened after the other view was taken back")
	}
	b.Park()
	b.Park() // pairs with one Unpark, not two
	b.Close()
	if !a.Fuse() {
		t.Fatal("Fuse refused after the parked view closed")
	}
	a.Unfuse()
	c := Shared([]int{0, 1, 2})
	defer c.Close()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("a view opened after a parked one closed is not counted")
	}
}
