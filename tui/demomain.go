package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/server/client"
	"github.com/muesli/termenv"
)

const demoUsage = `usage: tui demo relocate [-connect addr] [-model name] [-devices cuda:0,vulkan:1] [-ctx 512] [-tokens 260] [-prompt text] [-ignore-eos] [-auto]

Shows a model's blocks and their history moving between the host and the
devices of a running jitllmd while a reply streams.`

// runDemo is `tui demo ...`; the only demo is relocate.
func runDemo(args []string) int {
	if len(args) == 0 || args[0] != "relocate" {
		fmt.Fprintln(os.Stderr, demoUsage)
		return 2
	}
	fs := flag.NewFlagSet("demo relocate", flag.ContinueOnError)
	var opt demoOptions
	var devices string
	fs.StringVar(&opt.Addr, "connect", client.DefaultAddr, "the jitllmd to drive")
	fs.StringVar(&opt.Model, "model", "", "a loaded model's id or file name; loaded when it is not (with -devices)")
	fs.StringVar(&devices, "devices", "", "devices to load the model on, comma separated")
	fs.IntVar(&opt.MaxSeq, "ctx", 512, "context length of the demo's session")
	fs.IntVar(&opt.MaxTokens, "tokens", 260, "tokens in a reply")
	fs.StringVar(&opt.Prompt, "prompt", "", "what to ask (default: a long story)")
	fs.BoolVar(&opt.IgnoreEOS, "ignore-eos", false, "keep generating past the model's end of text")
	fs.BoolVar(&opt.Auto, "auto", false, "start the scripted run at once")
	if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, demoUsage)
		return 2
	}
	if devices != "" {
		opt.Devices = strings.Split(devices, ",")
	}
	conn, err := client.Discover(context.Background(), opt.Addr, localserver.WithAPIKey(os.Getenv("JITLLM_API_KEY")))
	if err != nil {
		fmt.Fprintf(os.Stderr, "no jitllmd answers at %s: %v\nstart one with: jitllmd serve -load <model>.jlm -devices cuda:0,vulkan:1\n", opt.Addr, err)
		return 1
	}
	d, err := newDemo(context.Background(), conn, opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	termenv.NewOutput(os.Stdout).SetBackgroundColor(termenv.RGBColor(string(cBg)))
	defer fmt.Print("\x1b]111\x07")
	p := tea.NewProgram(d, tea.WithAltScreen())
	// A closed terminal hangs up on us: the session still has to be closed, or
	// the blocks it holds cannot move for the next client.
	hup := make(chan os.Signal, 1)
	signal.Notify(hup, syscall.SIGHUP)
	go func() {
		<-hup
		p.Quit()
	}()
	if _, err := p.Run(); err != nil {
		d.close()
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	d.close()
	return 0
}
