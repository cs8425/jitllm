// Package client finds and dials a running jitllmd for a front end (the
// window, the terminal app): the address it defaults to, and ServerInfo,
// checked for compatibility before anything relies on it. The address, the
// models folder and the API version live in internal/localserver, which the
// root module's CLI shares; this package adds the generated ConnectRPC stubs,
// which only modules that depend on the server module can use.
package client

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/internal/localserver"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// DefaultAddr is where the local service listens unless configured otherwise.
const DefaultAddr = localserver.DefaultAddr

// Conn is a dialed jitllmd: every service's client on one base URL.
type Conn struct {
	Base      string
	Info      *v1.ServerInfo
	Model     jitllmv1connect.ModelServiceClient
	Device    jitllmv1connect.DeviceServiceClient
	Placement jitllmv1connect.PlacementServiceClient
	Session   jitllmv1connect.SessionServiceClient
	Inference jitllmv1connect.InferenceServiceClient
	Telemetry jitllmv1connect.TelemetryServiceClient
	// Legacy is a service found on localserver.LegacyAddr when the default
	// address was asked (see Discover).
	Legacy bool
}

// Dial makes the clients for addr (host:port, :port or a URL; "" is
// DefaultAddr) without touching the network. localserver.WithAPIKey makes
// every call carry the service's API key.
func Dial(addr string, opts ...localserver.Option) *Conn {
	if addr == "" {
		addr = DefaultAddr
	}
	base := localserver.BaseURL(addr)
	h := &http.Client{Transport: localserver.KeyTransport(localserver.APIKey(opts...), nil)}
	return &Conn{
		Base:      base,
		Model:     jitllmv1connect.NewModelServiceClient(h, base),
		Device:    jitllmv1connect.NewDeviceServiceClient(h, base),
		Placement: jitllmv1connect.NewPlacementServiceClient(h, base),
		Session:   jitllmv1connect.NewSessionServiceClient(h, base),
		Inference: jitllmv1connect.NewInferenceServiceClient(h, base),
		Telemetry: jitllmv1connect.NewTelemetryServiceClient(h, base),
	}
}

// Discover asks addr for its ServerInfo, within two seconds. An error that
// wraps localserver.ErrIncompatible is a jitllmd that answered and speaks
// another API version: say so and stop. Any other error is no jitllmd there,
// and a front end runs embedded. An error wrapping localserver.ErrUnauthorized
// is one that wants an API key.
//
// When addr is the default and nothing answers there, a jitllmd on the old
// default port (localserver.LegacyAddr) is used instead and the Conn says so
// in Legacy: a service installed before the default moved keeps its address.
func Discover(ctx context.Context, addr string, opts ...localserver.Option) (*Conn, error) {
	c, err := discoverAt(ctx, addr, opts)
	legacy, ok := localserver.LegacyFor(addr, opts...)
	if err == nil || !ok || errors.Is(err, localserver.ErrIncompatible) ||
		errors.Is(err, localserver.ErrUnauthorized) {
		return c, err
	}
	old, lerr := discoverAt(ctx, legacy, opts)
	if lerr == nil || errors.Is(lerr, localserver.ErrIncompatible) || errors.Is(lerr, localserver.ErrUnauthorized) {
		old.Legacy = true
		return old, lerr
	}
	return c, err
}

func discoverAt(ctx context.Context, addr string, opts []localserver.Option) (*Conn, error) {
	c := Dial(addr, opts...)
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	r, err := c.Telemetry.GetServerInfo(ctx, connect.NewRequest(&v1.GetServerInfoRequest{}))
	if err != nil {
		if connect.CodeOf(err) == connect.CodeUnauthenticated {
			return c, fmt.Errorf("%w at %s: give it the key (-api-key, or JITLLM_API_KEY)", localserver.ErrUnauthorized, addr)
		}
		return nil, err
	}
	c.Info = r.Msg.GetInfo()
	if err := localserver.Compatible(int(c.Info.GetApiVersion())); err != nil {
		return c, err
	}
	return c, nil
}

// Has reports whether the server names feature among its features.
func (c *Conn) Has(feature string) bool {
	return c.Info != nil && slices.Contains(c.Info.GetFeatures(), feature)
}

// Capabilities decodes the server's capability manifest (GET
// /v1/capabilities, internal/capabilities.Manifest) into v. It takes the
// caller's type so this thin client does not import the manifest's package,
// which reaches the converter.
func (c *Conn) Capabilities(v any) error {
	if c.Info == nil {
		return nil
	}
	return json.Unmarshal([]byte(c.Info.GetCapabilitiesJson()), v)
}
