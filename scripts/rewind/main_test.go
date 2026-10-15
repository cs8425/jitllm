package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type repo struct {
	t   *testing.T
	dir string
}

func (r *repo) git(date string, args ...string) string {
	r.t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = r.dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.com",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.com",
		"GIT_AUTHOR_DATE="+date, "GIT_COMMITTER_DATE="+date)
	out, err := cmd.CombinedOutput()
	if err != nil {
		r.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (r *repo) write(name string, lines []string) {
	r.t.Helper()
	p := filepath.Join(r.dir, name)
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		r.t.Fatal(err)
	}
}

func seq(prefix string, n int) []string {
	var s []string
	for i := 0; i < n; i++ {
		s = append(s, fmt.Sprintf("%s %d", prefix, i))
	}
	return s
}

const (
	t0 = "2000-01-01T00:00:00Z"
	t1 = "2000-01-02T00:00:00Z" // the PR is written
	t2 = "2000-01-03T00:00:00Z" // another PR lands
	t3 = "2000-01-04T00:00:00Z"
)

// fixture builds main with a file a, then a feature PR (lines at the end of
// a, plus a new file f.go) merged after t1, and returns the repo and the old
// main.
func fixture(t *testing.T) (r *repo, oldMain string) {
	r = &repo{t: t, dir: t.TempDir()}
	r.git(t0, "init", "-q", "-b", "main")
	r.write("a.txt", seq("base", 40))
	r.git(t0, "add", "-A")
	r.git(t0, "commit", "-qm", "init")
	oldMain = r.git(t0, "rev-parse", "HEAD")
	r.git(t0, "checkout", "-q", "-b", "feature")
	r.write("a.txt", append(seq("base", 40), seq("feature", 10)...))
	r.write("f.go", seq("package feature", 12))
	r.git(t2, "add", "-A")
	r.git(t2, "commit", "-qm", "feature work")
	r.git(t2, "checkout", "-q", "main")
	r.git(t2, "merge", "-q", "--no-ff", "-m", "Merge pull request #1 from x/feature", "feature")
	return
}

func run(t *testing.T, r *repo, head, body string) []finding {
	t.Helper()
	fs, err := check(r.dir, "main", head, body, 3)
	if err != nil {
		t.Fatal(err)
	}
	return fs
}

// The incident: a commit written at t1 against the old main, committed on the
// new one, carrying the old a.txt and without f.go.
func rewindCommit(r *repo, oldMain string) string {
	r.git(t1, "checkout", "-q", "-b", "pr")
	r.git(t1, "checkout", oldMain, "--", "a.txt")
	r.git(t1, "rm", "-q", "f.go")
	r.write("new.txt", seq("mine", 5))
	r.git(t1, "add", "-A")
	r.git(t1, "commit", "-qm", "my work")
	return r.git(t1, "rev-parse", "HEAD")
}

func TestARewoundCommitFails(t *testing.T) {
	r, oldMain := fixture(t)
	fs := run(t, r, rewindCommit(r, oldMain), "")
	got := map[string]finding{}
	for _, f := range fs {
		got[f.Path] = f
	}
	for _, p := range []string{"a.txt", "f.go"} {
		f, ok := got[p]
		if !ok {
			t.Fatalf("%s not reported: %+v", p, fs)
		}
		if !strings.Contains(f.Commit, "feature work") || !strings.Contains(f.Merge, "pull request #1") {
			t.Errorf("%s blames %q via %q", p, f.Commit, f.Merge)
		}
	}
	if len(got) != 2 {
		t.Errorf("unexpected findings: %+v", fs)
	}
}

func TestAnOrdinaryEditPasses(t *testing.T) {
	r, oldMain := fixture(t)
	// Written at t1 against the old main, editing the start of a.txt while the
	// feature edited its end: the merge base is the old main.
	r.git(t1, "checkout", "-q", "-b", "pr", oldMain)
	lines := seq("base", 40)
	lines[2] = "edited by the PR"
	lines[3] = "edited by the PR"
	lines[4] = "edited by the PR"
	r.write("a.txt", lines)
	r.git(t1, "add", "-A")
	r.git(t1, "commit", "-qm", "edit")
	if fs := run(t, r, "HEAD", ""); len(fs) != 0 {
		t.Fatalf("flagged an ordinary PR: %+v", fs)
	}
	// The same edit rebased onto the new main.
	r.git(t3, "rebase", "-q", "main")
	if fs := run(t, r, "HEAD", ""); len(fs) != 0 {
		t.Fatalf("flagged a rebased PR: %+v", fs)
	}
}

func TestADeclaredRemovalPasses(t *testing.T) {
	r, oldMain := fixture(t)
	head := rewindCommit(r, oldMain)
	body := "Cleanup.\n\nRemoves: a.txt (superseded)\nRemoves: f.go (moved)\n"
	if fs := run(t, r, head, body); len(fs) != 0 {
		t.Fatalf("declared removals flagged: %+v", fs)
	}
	if fs := run(t, r, head, "Removes: f.go (moved)"); len(fs) != 1 || fs[0].Path != "a.txt" {
		t.Fatalf("a partial declaration must still flag the rest: %+v", fs)
	}
}

func TestRemovesLines(t *testing.T) {
	got := removes("x\nRemoves: a/b.go (why)\n  Removes: no\nRemoves: dir/\n")
	if strings.Join(got, ",") != "a/b.go,dir/" {
		t.Fatalf("got %v", got)
	}
	if !covered([]string{"dir/"}, "dir/x.go") || covered([]string{"dir"}, "dir/x.go") {
		t.Fatal("prefix rule")
	}
}

// The CI job runs main's copy of the program against a PR checkout that has
// no scripts/rewind of its own (a branch older than the check): -C names the
// repository, and nothing is read from it but git's objects.
func TestRunsAgainstARepositoryWithoutTheProgram(t *testing.T) {
	r, oldMain := fixture(t)
	head := rewindCommit(r, oldMain)
	if _, err := os.Stat(filepath.Join(r.dir, "scripts", "rewind")); err == nil {
		t.Fatal("fixture carries the program")
	}
	cmd := exec.Command("go", "run", ".", "-C", r.dir, "-base", "main", "-head", head)
	out, err := cmd.CombinedOutput()
	if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
		t.Fatalf("want exit 1, got %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "a.txt") {
		t.Fatalf("a.txt not reported:\n%s", out)
	}
}
