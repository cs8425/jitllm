package server

import (
	"errors"
	"net"
	"strings"
	"syscall"
)

// Listen opens the listeners for addr. The name "localhost" is bound to its
// loopback addresses, 127.0.0.1 and [::1], one listener each, so a client that
// resolves localhost to either family reaches the server and nothing off this
// machine does. The IPv6 listener is skipped where the host has no IPv6
// loopback; IPv4 is required. Any other address is one listener, as net.Listen
// has it (":4242" is every interface, and is the caller's choice to make).
//
// With port 0 the IPv4 listener picks the port and the IPv6 one takes the same.
func Listen(addr string) ([]net.Listener, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil || !strings.EqualFold(host, "localhost") {
		ln, err := net.Listen("tcp", addr)
		if err != nil {
			return nil, err
		}
		return []net.Listener{ln}, nil
	}
	v4, err := net.Listen("tcp4", net.JoinHostPort("127.0.0.1", port))
	if err != nil {
		return nil, err
	}
	lns := []net.Listener{v4}
	if _, p, err := net.SplitHostPort(v4.Addr().String()); err == nil {
		port = p
	}
	v6, err := net.Listen("tcp6", net.JoinHostPort("::1", port))
	switch {
	case err == nil:
		lns = append(lns, v6)
	case errors.Is(err, syscall.EADDRINUSE):
		v4.Close()
		return nil, err
	}
	return lns, nil
}
