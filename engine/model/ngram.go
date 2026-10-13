package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The n-gram embedding: Qwen3.8-Flash-Next's PLE (transformers'
// Qwen4ExpTextPLELayer, llama.cpp's build_ple), on one linear block, before
// its attention mix.
//
// Each position's token and its NGram-1 predecessors are hashed per head
// (Config.NGramMul, NGramOff, NGramVoc; a predecessor at or before an
// end-of-sequence token, or before the sequence, reads as that token) into a
// row of a shared table, which is read a row at a time from the container
// (jlm.RowTable): it is hundreds of millions of rows, of which a position
// reads (NGram-1)*NGramHeadsPer. The rows, concatenated, project to a key
// per stream and one value; each stream's normed key and normed residual give
// a gate, sigmoid of the signed square root of their scaled dot product; the
// gated value is added to the stream, and so is a depthwise convolution over
// it (normed), dilated by NGram, through silu.
//
// The predecessors and the convolution's history are the block's recurrent
// state, after the delta rule's (Config.ngramStateLen), so they page,
// relocate and are prefix-cached with it.

// ngramTable is the container's n-gram table.
type ngramTable struct {
	f        *jlm.File
	e        *jlm.Entry
	typ      quant.Type
	rowBytes int
}

// ngramScratch is a State's n-gram buffers.
type ngramScratch struct {
	rows int
	// per row: the hashed table rows, the gathered embedding, its key and
	// value, the normed residual, the gate per stream and the gated value
	// (normed in place for the convolution once added).
	idx                         []uint64
	emb, key, val, q, gate, gv  []float32
	gvn, conv, tap              []float32
	win                         []int32
	raw, scr                    []byte
	dot                         *nn.AttnSet
}

// allocNGram sizes the n-gram buffers for one row.
func (s *State) allocNGram() {
	c := s.m.Cfg
	if c.NGramBlock == 0 {
		return
	}
	g := &ngramScratch{}
	s.q4.ng = g
	g.idx = make([]uint64, c.ngramHeads())
	g.win = make([]int32, c.NGram)
	g.raw = make([]byte, s.m.ngram.rowBytes)
	g.scr = jlm.RowScratch()
	hd := c.HCMult * c.NEmbd
	g.conv, g.tap = make([]float32, hd), make([]float32, hd)
	g.dot = s.attnSetF32(c.NEmbd)
}

// growNGram sizes the per-row buffers for n rows.
func (s *State) growNGram(n int) {
	c := s.m.Cfg
	g := s.q4.ng
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	hd := c.HCMult * c.NEmbd
	g.emb = make([]float32, n*c.NGramWidth())
	g.key, g.q, g.gv, g.gvn = make([]float32, n*hd), make([]float32, n*hd), make([]float32, n*hd), make([]float32, n*hd)
	g.val = make([]float32, n*c.NEmbd)
	g.gate = make([]float32, n*c.HCMult)
}

// ngramState is sequence slot's n-gram state in block li: its predecessors
// and its convolution history, after every slot's delta-rule state.
func (s *State) ngramState(li, slot int) (prev, hist []float32) {
	c := s.m.Cfg
	n := c.ngramStateLen()
	nseq := len(s.rstate[li]) / (s.dg.deltaStateLen() + n)
	base := nseq * s.dg.deltaStateLen()
	st := s.rstate[li][base+slot*n : base+(slot+1)*n]
	return st[:c.NGram-1], st[c.NGram-1:]
}

// ngramApply adds the n-gram embedding into n rows of x in block li. Row r is
// sequence slot(r)'s next position with token tok(r) (-1 for a row with no
// token, which hashes as the picture placeholder); rows of one slot are in
// order.
func (s *State) ngramApply(li int, l *layer, x []float32, n int, slot func(r int) int, tok func(r int) int32) error {
	c, g, d := s.m.Cfg, s.q4.ng, l.q4e
	if c.q4eFault == q4eFaultNoNGram {
		return nil
	}
	h, e := c.HCMult, c.NEmbd
	hd, w, rd := h*e, c.NGramWidth(), c.NGramRowDim
	for r := 0; r < n; r++ {
		prev, _ := s.ngramState(li, slot(r))
		t := tok(r)
		if t < 0 {
			t = c.NGramImage
		}
		ngramWindow(g.win, prev, t, c.NGramEOS, c.q4eFault != q4eFaultNoEOSReset)
		ngramHash(g.idx, g.win, c.NGramMul, c.NGramOff, c.NGramVoc, c.NGramHeadsPer)
		// The predecessors move up one, stored as id+1.
		copy(prev[1:], prev[:len(prev)-1])
		prev[0] = float32(t) + 1
		for hh, row := range g.idx {
			if err := s.m.ngram.f.ReadRow(s.m.ngram.e, row, g.raw, g.scr); err != nil {
				return err
			}
			if err := ngramRow(g.emb[r*w+hh*rd:r*w+(hh+1)*rd], s.m.ngram.typ, g.raw); err != nil {
				return err
			}
		}
	}
	s.jit.NewInput()
	if err := s.mm(g.key[:n*hd], d.ngKey, g.emb[:n*w], n); err != nil {
		return err
	}
	if err := s.mm(g.val[:n*e], d.ngVal, g.emb[:n*w], n); err != nil {
		return err
	}
	// The key per stream, normed in place, and the residual's streams normed.
	for i := 0; i < n*h; i++ {
		k := g.key[i*e : (i+1)*e]
		s.rmsnorm(k, k, d.ngNormK[(i%h)*e:(i%h+1)*e], c.RMSEps)
	}
	s.hcNorm(g.q[:n*hd], x, n, d.ngNormQ)
	s.jit.NewInput()
	for i := 0; i < n*h; i++ {
		g.dot.AttnScores(g.gate[i:i+1], g.key[i*e:(i+1)*e], g.q[i*e:(i+1)*e], 1)
	}
	gate := g.gate[:n*h]
	s.scale(gate, invSqrt(e))
	ngramGate(gate)
	s.jit.NewInput()
	for i := 0; i < n*h; i++ {
		v := g.gv[i*e : (i+1)*e]
		copy(v, g.val[(i/h)*e:(i/h+1)*e])
		s.scale(v, gate[i])
		s.rmsnorm(g.gvn[i*e:(i+1)*e], v, d.ngNormC[(i%h)*e:(i%h+1)*e], c.RMSEps)
	}
	s.jit.NewInput()
	hist := c.ngramHist()
	for r := 0; r < n; r++ {
		_, hs := s.ngramState(li, slot(r))
		cur := g.gvn[r*hd : (r+1)*hd]
		// Tap t reads (NGramConv-1-t)*NGram positions back: the last tap is
		// the row itself, the others the history, oldest first.
		clear(g.conv)
		for t := 0; t < c.NGramConv; t++ {
			back := (c.NGramConv - 1 - t) * c.NGram
			src := cur
			if back > 0 {
				p := hist - back
				src = hs[p*hd : (p+1)*hd]
			}
			copy(g.tap, src)
			nn.ActMul32JIT(g.tap, d.ngCv[t*hd:(t+1)*hd], nn.ActIdentity)
			s.axpy(g.conv, g.tap, 1)
		}
		nn.Act32JIT(g.conv, nn.ActSiLU)
		s.jit.NewInput()
		if hist > 0 {
			copy(hs, hs[hd:])
			copy(hs[(hist-1)*hd:], cur)
		}
		for k := 0; k < h; k++ {
			xk := c.stream(x, k, n, r)
			s.axpy(xk, g.gv[(r*h+k)*e:(r*h+k+1)*e], 1)
			if c.q4eFault != q4eFaultNoNGramConv {
				s.axpy(xk, g.conv[k*e:(k+1)*e], 1)
			}
		}
	}
	s.jit.NewInput()
	return nil
}

// ngramRow decodes one table row into dst.
func ngramRow(dst []float32, t quant.Type, raw []byte) error {
	switch t {
	case quant.F32, quant.F16, quant.BF16:
		nn.Row32JIT(dst, t, raw, 0, len(dst))
		return nil
	}
	return fmt.Errorf("model: an n-gram table row of %v", t)
}

// ngramWindow fills win with the token t and its predecessors (prev, stored
// as id+1, zero for none): a predecessor at or before an end-of-sequence
// token, or before the sequence, reads as eos. The token's own end of
// sequence does not cut its window.
//
// TEMPORARY: Go integer logic, to be replaced by the generated hash kernel.
func ngramWindow(win []int32, prev []float32, t, eos int32, reset bool) {
	win[0] = t
	cut := false
	for s := 1; s < len(win); s++ {
		v := int32(prev[s-1]) - 1
		if v < 0 {
			v = eos
		}
		if cut {
			v = eos
		}
		if reset && v == eos {
			cut = true
		}
		win[s] = v
	}
}

// ngramHash writes each head's table row for the window.
//
// TEMPORARY: Go integer arithmetic, to be replaced by the generated hash
// kernel.
func ngramHash(idx []uint64, win []int32, mul, off, voc []uint64, per int) {
	for n := 2; n <= len(win); n++ {
		mixed := uint64(win[0]) * mul[0]
		for j := 1; j < n; j++ {
			mixed ^= uint64(win[j]) * mul[j]
		}
		for k := 0; k < per; k++ {
			h := (n-2)*per + k
			idx[h] = mixed%voc[h] + off[h]
		}
	}
}

// ngramGate is sigmoid(sign(s) * sqrt(max(|s|, 1e-6))) in place.
//
// TEMPORARY: Go arithmetic, to be replaced by a generated kernel.
func ngramGate(s []float32) {
	for i, v := range s {
		m := math.Sqrt(math.Max(math.Abs(float64(v)), 1e-6))
		if v < 0 {
			m = -m
		}
		s[i] = float32(1 / (1 + math.Exp(-m)))
	}
}
