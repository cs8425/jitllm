//go:build !darwin

package sched

// EQoS sets nothing off macOS, where a pool's cores are its process mask's
// (RULE 5), and reports false.
func EQoS() bool { return false }

// allE is false off macOS: no pool asks for a core class here.
func allE([]int) bool { return false }
