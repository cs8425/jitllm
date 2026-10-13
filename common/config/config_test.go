package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestKeysFallBackToTheEnvironment(t *testing.T) {
	defer UseEnv(Env{})
	UseEnv(Env{APIKey: "from-env"})
	c := &Config{}
	if c.ServingKey() != "from-env" || c.ServiceAPIKey() != "from-env" {
		t.Errorf("env: %q %q", c.ServingKey(), c.ServiceAPIKey())
	}
	c = &Config{APIKey: "a", ServiceKey: "s"}
	if c.ServingKey() != "a" || c.ServiceAPIKey() != "s" {
		t.Errorf("file: %q %q", c.ServingKey(), c.ServiceAPIKey())
	}
	var none *Config
	if none.ServingKey() != "from-env" {
		t.Errorf("a nil config: %q", none.ServingKey())
	}
	UseEnv(Env{})
	if (&Config{}).ServingKey() != "" {
		t.Error("a key out of nothing")
	}
}

// TestASavedLegacyAPIAddressFollowsTheDefault: a settings file written before
// the default moved holds the old default, which the app wrote and the person
// never chose; any other address is theirs.
func TestASavedLegacyAPIAddressFollowsTheDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("HOME", dir)
	p := ConfigPath()
	if p == "" {
		t.Fatal("this system has no config folder, so LoadConfig reads nothing and the gate proves nothing")
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	for saved, want := range map[string]string{
		"127.0.0.1:8080": "localhost:4242",
		"127.0.0.1:9000": "127.0.0.1:9000",
	} {
		if err := os.WriteFile(p, []byte(`{"api_addr":"`+saved+`"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := LoadConfig().APIAddr; got != want {
			t.Errorf("saved %q loads as %q, want %q", saved, got, want)
		}
	}
}
