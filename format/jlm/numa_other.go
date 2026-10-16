//go:build !linux

package jlm

// placeMemory has no memory policy or huge pages to ask for here.
func placeMemory(b []byte, noHuge bool, mask []uint64) {}

// bindNode has no memory policy to set here.
func bindNode(b []byte, mask []uint64) {}
