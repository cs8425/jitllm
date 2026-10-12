//go:build !jitllmtest

package model

// persistRan is the jitllmtest build's check (persistdefault_test.go); here it
// passes the code through.
func persistRan(code int) int { return code }
