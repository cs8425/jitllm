//go:build jitllmtest

package model

import (
	"os"
	"strconv"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// JITLLM_TEST_GEMM_DEEP=1 runs every model gate with the deep m16n8 GEMM
// tiles forced on (tier.ForceGemmDeepForTest), -1 forced off.
func init() {
	if n, err := strconv.Atoi(os.Getenv("JITLLM_TEST_GEMM_DEEP")); err == nil {
		tier.ForceGemmDeepForTest(n)
	}
}
