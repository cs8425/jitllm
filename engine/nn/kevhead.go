package nn

// KevHead is Kev's pointer head as a device reads it (engine/model's
// decidekev.go): the output RMSNorm, cls.output split in its query half (WQ,
// BQ; P x D and P) and its key half (WK, BK), f32.
type KevHead struct {
	D, P    int
	RMSEps  float32
	OutNorm []float32
	WQ, WK  []float32
	BQ, BK  []float32
}

// KevHeadDevice is a device that runs Kev's head: q is the pointer row's
// final residual (D), k the option rows' (n x D), and out gets n scores, the
// query's projection dotted with each key's over sqrt(P). It returns false
// having written nothing it should be read for, and the caller runs the head
// on the host.
type KevHeadDevice interface {
	KevHead(h *KevHead, q, k, out []float32) bool
}
