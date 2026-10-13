package server

import (
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/localserver"
)

func ipv6Loopback() bool {
	l, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		return false
	}
	l.Close()
	return true
}

// TestTheDefaultListensOnLoopbackOnly: the default address binds 127.0.0.1 and,
// where the host has it, [::1]; never an unspecified address; and an address
// of this machine's other interfaces does not reach it.
func TestTheDefaultListensOnLoopbackOnly(t *testing.T) {
	host, port, err := net.SplitHostPort(localserver.DefaultAddr)
	if err != nil || host != "localhost" || port != "4242" {
		t.Fatalf("DefaultAddr = %q", localserver.DefaultAddr)
	}
	if !Loopback(localserver.DefaultAddr) {
		t.Fatal("the default address is not loopback")
	}

	// The default's host on a free port: 4242 may be taken on this machine.
	lns, err := Listen("localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, l := range lns {
			l.Close()
		}
	}()
	var v4, v6 bool
	var port0 string
	for _, l := range lns {
		a := l.Addr().(*net.TCPAddr)
		if !a.IP.IsLoopback() || a.IP.IsUnspecified() {
			t.Fatalf("listening on %s, which is not loopback", a)
		}
		if a.IP.To4() != nil {
			v4 = true
		} else {
			v6 = true
		}
		if port0 == "" {
			port0 = strconv.Itoa(a.Port)
		} else if port0 != strconv.Itoa(a.Port) {
			t.Errorf("the families listen on different ports: %v", lns)
		}
	}
	if !v4 {
		t.Error("no IPv4 loopback listener")
	}
	if ipv6Loopback() && !v6 {
		t.Error("this host has an IPv6 loopback and the default did not listen on it")
	}

	// Both families answer on "localhost"'s addresses.
	for _, l := range lns {
		go func() {
			if c, err := l.Accept(); err == nil {
				c.Close()
			}
		}()
		c, err := net.DialTimeout("tcp", l.Addr().String(), 2*time.Second)
		if err != nil {
			t.Errorf("dialling %s: %v", l.Addr(), err)
			continue
		}
		c.Close()
	}

	// Another interface of this machine is refused. A machine with only
	// loopback has none to try, which the listener addresses above already
	// cover.
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	tried := 0
	for _, a := range addrs {
		ip, ok := a.(*net.IPNet)
		if !ok || ip.IP.IsLoopback() || ip.IP.IsLinkLocalUnicast() {
			continue
		}
		c, err := net.DialTimeout("tcp", net.JoinHostPort(ip.IP.String(), port0), time.Second)
		if err == nil {
			c.Close()
			t.Errorf("%s reached the default listener", ip.IP)
		}
		tried++
	}
	t.Logf("tried %d non-loopback addresses of this machine", tried)
}

// TestListenKeepsWhatItIsToldElsewhere: only the name localhost is widened to
// both families; an explicit address is one listener, and ":0" is the
// caller's choice and not the default.
func TestListenKeepsWhatItIsToldElsewhere(t *testing.T) {
	lns, err := Listen("127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer lns[0].Close()
	if len(lns) != 1 || lns[0].Addr().(*net.TCPAddr).IP.String() != "127.0.0.1" {
		t.Fatalf("127.0.0.1 gave %v", lns)
	}
}
