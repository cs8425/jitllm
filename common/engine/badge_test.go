package engine

import "testing"

func TestBadgeSaysWhichEngine(t *testing.T) {
	for _, c := range []struct {
		w    Where
		api  string
		want string
	}{
		{Where{}, "", "Embedded · API off"},
		{Where{}, "localhost:4242", "Embedded · API on :4242"},
		{Where{}, "0.0.0.0:9000", "Embedded · API on 0.0.0.0:9000"},
		{Where{Service: true, Addr: "localhost:4242", Version: "0.9.0"}, "", "Service · localhost:4242 (jitllmd 0.9.0)"},
		{Where{Service: true, Addr: "localhost:4242", Version: "0.9.0", Lost: true}, "", "Service · not answering · localhost:4242 (jitllmd 0.9.0)"},
	} {
		if got := Badge(c.w, c.api); got != c.want {
			t.Errorf("Badge(%+v, %q) = %q, want %q", c.w, c.api, got, c.want)
		}
	}
}
