// Package api serves jitllm's API (server.Engine.Handler: ConnectRPC, the
// OpenAI and Anthropic shims) from inside a front end, so the window and the
// terminal can turn it on without running jitllmd beside them.
//
// It serves the front end's own engine (common/engine.Engine.Server): a model
// open in the app is a model of the API, and a request through the API is one
// more session on it, under the same budget and gates.
package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/server"
)

// DefaultAddr is loopback only: turning the API on must not open it to the
// network unless the address says so.
const DefaultAddr = localserver.DefaultAddr

// Option configures a Server.
type Option func(*settings)

type settings struct {
	key     string
	origins []string
}

// WithKey makes every route but /healthz require key as a bearer token.
func WithKey(key string) Option { return func(s *settings) { s.key = key } }

// WithOrigins lets these browser origins call the API cross-origin.
func WithOrigins(origins ...string) Option {
	return func(s *settings) { s.origins = append(s.origins, origins...) }
}

// Server is a running API.
type Server struct {
	srv  *http.Server
	addr string
	// asked is the address as the setting spells it, so Toggle can tell a
	// changed setting from the one already served.
	asked string
	// key is the API key it requires, so a changed key restarts it.
	key string
}

// Start listens on addr and serves e. The listen happens before it returns,
// so an address in use is its error.
//
// The API is guarded as jitllmd's is (server.Guard): a loopback listener
// refuses a Host header that is not its own, a browser origin that is not
// loopback or listed, and, with WithKey, a request without the key.
func Start(addr string, e *server.Engine, opts ...Option) (*Server, error) {
	if addr == "" {
		addr = DefaultAddr
	}
	var o settings
	for _, f := range opts {
		f(&o)
	}
	lns, err := server.Listen(addr)
	if err != nil {
		return nil, err
	}
	g := server.Guard{APIKey: o.key, Addr: addr, AllowOrigins: o.origins}
	s := &Server{addr: lns[0].Addr().String(), asked: addr, key: o.key, srv: &http.Server{
		Handler: g.Wrap(e.Handler()),
		// No WriteTimeout, as jitllmd: a streaming generate can run for minutes.
		ReadHeaderTimeout: 20 * time.Second,
	}}
	for _, ln := range lns {
		go func() {
			if err := s.srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
				ln.Close()
			}
		}()
	}
	return s, nil
}

// Addr is the address it listens on.
func (s *Server) Addr() string { return s.addr }

// Close stops listening and waits for the requests in flight. The engine and
// its models stay: they are the front end's.
func (s *Server) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s.srv.Shutdown(ctx)
}

// Toggle brings cur in line with a setting: off closes it, on starts one, and
// on with a different address restarts it. It returns what now runs (nil when
// off) and a line for the front end's status.
//
// A jitllmd already answering at the address refuses the start: two engines
// on one machine share its GPU memory, and the app would be a second engine
// nobody asked for. held is that service, so the front end can offer to
// connect to it instead; it is the zero Info otherwise.
func Toggle(cur *Server, on bool, addr string, e *server.Engine, opts ...Option) (s *Server, msg string, held localserver.Info) {
	if addr == "" {
		addr = DefaultAddr
	}
	var o settings
	for _, f := range opts {
		f(&o)
	}
	if cur != nil && on && cur.asked == addr && cur.key == o.key {
		return cur, "API on http://" + cur.Addr(), held
	}
	if cur != nil {
		cur.Close()
	}
	if !on {
		if cur == nil {
			return nil, "", held
		}
		return nil, "API off", held
	}
	if in, err := localserver.Probe(context.Background(), addr, localserver.WithAPIKey(o.key)); err == nil ||
		errors.Is(err, localserver.ErrIncompatible) || errors.Is(err, localserver.ErrUnauthorized) {
		if in.Addr == "" {
			in.Addr = addr
		}
		return nil, Refusal(in), in
	}
	s, err := Start(addr, e, opts...)
	if err != nil {
		return nil, "API not started: " + err.Error(), held
	}
	return s, "API on http://" + s.Addr(), held
}

// Refusal is the line that says why the API did not start beside in.
func Refusal(in localserver.Info) string {
	v := ""
	if in.Version != "" {
		v = " " + in.Version
	}
	return "API not started: jitllmd" + v + " already serves " + in.Addr + "; connect to it instead"
}
