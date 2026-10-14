package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// BitNet b1.58 (jlm.ArchBitNet). Its seven projections are ternary, and every
// source form of them reaches the container as TypeTQ2 -- GGUF's TQ2_0 blocks
// -- through one of the conversions here, each exact or refused:
//
//	TQ2_0 (llama.cpp --outtype tq2_0)          carried
//	TQ1_0 (llama.cpp --outtype tq1_0)          its base-3 digits re-packed as 2-bit codes
//	I2_S (bitnet.cpp)                          its codes re-packed, its f32 scale as the
//	                                           blocks' f16 d when that is exact
//	F32/F16/BF16 holding a ternary tensor      the codes and the magnitude read off it
//	(llama.cpp's converter dequantizes)
//	the HuggingFace release, uint8 codes       unpacked, weight_scale as d
//	plus a bf16 weight_scale (offline)
//	the bf16 latent checkpoint (online)        BitNetForCausalLM's own absmean
//	                                           ternarization, in bf16 as the class runs it
//
// The graph's other facts -- squared ReLU on the gate, NEOX rotary on the
// rows as shipped, the two inner norms, per-token int8 activations -- are the
// arch's (bitnetConfig, jlm.ArchBitNet).

// bitnetProjections are the roles a BitNet block runs as BitLinear.
var bitnetProjections = map[jlm.Role]bool{
	jlm.RoleAttnQ: true, jlm.RoleAttnK: true, jlm.RoleAttnV: true, jlm.RoleAttnOut: true,
	jlm.RoleFFNGate: true, jlm.RoleFFNUp: true, jlm.RoleFFNDown: true,
}

// bitnetConfig is the arch's graph on the GGUF side. Neither llama.cpp's
// "bitnet" nor bitnet.cpp's "bitnet-b1.58" states an activation; the class
// both convert, BitNetForCausalLM, declares relu2 (transformers'
// BitNetConfig.hidden_act) and rotates with rotate_half over the rows as
// shipped, which neither converter permutes. llama.cpp runs SiLU and its
// NORM pairs here, which is not what the weights were trained under (RULE 7m).
func bitnetConfig(c *jlm.Config) {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagReLU2
}

// tq2Put writes one TQ2_0 block: 256 codes (weight+1, in 0..2) and the f16
// scale. Element 128*(j/32) + 32*l + j%32 is bits 2l of byte j (quant's
// decoder is the specification).
func tq2Put(dst []byte, code func(e int) uint8, d uint16) {
	for j := 0; j < 64; j++ {
		base := 128*(j/32) + j%32
		var b byte
		for l := 0; l < 4; l++ {
			b |= code(base+32*l) << (2 * l)
		}
		dst[j] = b
	}
	binary.LittleEndian.PutUint16(dst[64:], d)
}

// exactHalf is v as an f16, and false when no f16 is exactly v.
func exactHalf(v float32) (uint16, bool) {
	h := quant.EncodeHalf(v)
	return h, float32(quant.DecodeHalf(h)) == v
}

// tq1ToTQ2 re-packs TQ1_0 blocks as TQ2_0: the same codes and the same d, so
// every weight decodes to the same value.
func tq1ToTQ2(src []byte, n uint64) ([]byte, error) {
	blocks := n / 256
	if n%256 != 0 || uint64(len(src)) < blocks*54 {
		return nil, fmt.Errorf("%d TQ1_0 elements in %d bytes", n, len(src))
	}
	out := make([]byte, blocks*66)
	vals := make([]float32, 256)
	for b := uint64(0); b < blocks; b++ {
		blk := src[b*54 : b*54+54]
		d := binary.LittleEndian.Uint16(blk[52:])
		// The digits through quant's own decoder at d = 1, so the order is
		// defined once.
		one := append(append([]byte(nil), blk[:52]...), 0x00, 0x3C)
		if err := quant.Dequant32(quant.TQ1_0, one, vals); err != nil {
			return nil, err
		}
		tq2Put(out[b*66:], func(e int) uint8 { return uint8(int(vals[e]) + 1) }, d)
	}
	return out, nil
}

// i2sToTQ2 re-packs bitnet.cpp's I2_S, the layout its x86 quantizer writes
// (ACT_PARALLEL, QK_I2_S 128, the layout of Microsoft's published GGUF):
// n/4 code bytes over the whole tensor, element 128i+32g+p in bits 6-2g of
// byte 32i+p, codes 0, 1, 2 for -1, 0, 1; then one f32 scale. The scale
// becomes every block's d, which is exact only when it is an f16; otherwise
// the tensor is refused rather than rounded.
func i2sToTQ2(src []byte, n uint64) ([]byte, error) {
	if n%256 != 0 || uint64(len(src)) < n/4+4 {
		return nil, fmt.Errorf("%d I2_S elements in %d bytes", n, len(src))
	}
	s := math.Float32frombits(binary.LittleEndian.Uint32(src[n/4:]))
	d, ok := exactHalf(s)
	if !ok {
		return nil, fmt.Errorf("its scale %v is not an f16 (nearest %v), and a ternary weight's "+
			"scale is carried as one: %w", s, quant.DecodeHalf(d), ErrNotImplemented)
	}
	blocks := n / 256
	out := make([]byte, blocks*66)
	for b := uint64(0); b < blocks; b++ {
		base := b * 256
		var bad error
		tq2Put(out[b*66:], func(e int) uint8 {
			i := base + uint64(e)
			c := src[(i/128)*32+i%32] >> (6 - 2*((i%128)/32)) & 3
			if c == 3 && bad == nil {
				bad = fmt.Errorf("element %d has code 3, which I2_S does not define", i)
			}
			return c
		}, d)
		if bad != nil {
			return nil, bad
		}
	}
	return out, nil
}

// ternaryFromFloats packs a tensor whose values are ternary -- every nonzero
// value of a 256-block the same magnitude -- as TQ2_0, each block's magnitude
// its d. llama.cpp's converter writes BitNet's projections this way (the
// ternarized weight, dequantized). A value that is not one of a block's three,
// or a magnitude that is not an f16, is refused: the tensor is not ternary.
func ternaryFromFloats(vals []float32) ([]byte, error) {
	if len(vals)%256 != 0 {
		return nil, fmt.Errorf("%d values is not a whole number of 256-blocks", len(vals))
	}
	out := make([]byte, len(vals)/256*66)
	for b := 0; b < len(vals)/256; b++ {
		v := vals[b*256 : b*256+256]
		var s float32
		for _, x := range v {
			if a := float32(math.Abs(float64(x))); a != 0 {
				if s == 0 {
					s = a
				} else if a != s {
					return nil, fmt.Errorf("block %d holds magnitudes %v and %v: not ternary", b, s, a)
				}
			}
		}
		d, ok := exactHalf(s)
		if !ok {
			return nil, fmt.Errorf("block %d's magnitude %v is not an f16", b, s)
		}
		tq2Put(out[b*66:], func(e int) uint8 {
			switch {
			case v[e] > 0:
				return 2
			case v[e] < 0:
				return 0
			}
			return 1
		}, d)
	}
	return out, nil
}

// floatsOf decodes a verbatim float tensor.
func floatsOf(t jlm.Type, data []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	switch t {
	case jlm.TypeF32:
		if len(data) < 4*n {
			return nil, fmt.Errorf("%d f32 values in %d bytes", n, len(data))
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
	case jlm.TypeF16, jlm.TypeBF16:
		if len(data) < 2*n {
			return nil, fmt.Errorf("%d 16-bit values in %d bytes", n, len(data))
		}
		for i := range out {
			h := binary.LittleEndian.Uint16(data[2*i:])
			if t == jlm.TypeF16 {
				out[i] = float32(quant.DecodeHalf(h))
			} else {
				out[i] = math.Float32frombits(uint32(h) << 16)
			}
		}
	default:
		return nil, fmt.Errorf("%v is not a float type", t)
	}
	return out, nil
}

// ternaryGGUF re-packs a GGUF tensor of a ternary source type as TQ2_0, and
// reports false for any other type.
func ternaryGGUF(t quant.Type, data []byte, n uint64) ([]byte, bool, error) {
	switch t {
	case quant.TQ1_0:
		b, err := tq1ToTQ2(data, n)
		return b, true, err
	case quant.I2_S:
		b, err := i2sToTQ2(data, n)
		return b, true, err
	}
	return nil, false, nil
}

// bitnetScaleOf reports a GGUF's optional per-tensor projection scale
// (llama.cpp's "blk.N.attn_q.scale"): the tensor name it scales.
func bitnetScaleOf(name string) (string, bool) {
	if !strings.HasPrefix(name, "blk.") || !strings.HasSuffix(name, ".scale") {
		return "", false
	}
	return strings.TrimSuffix(name, ".scale") + ".weight", true
}

// bitnetTensors finishes a BitNet source: every projection becomes TypeTQ2,
// its optional per-tensor scale folded into each block's d (refused where the
// product is not an f16), and anything that is not ternary is refused by name.
// scales maps a weight's name to its scale.
func bitnetTensors(s *jlm.Source, scales map[string]float32) error {
	if s.Config == nil || s.Config.Arch != jlm.ArchBitNet {
		if len(scales) > 0 {
			return fmt.Errorf("convert: %d projection scales on a %v model", len(scales), s.Config.Arch)
		}
		return nil
	}
	used := 0
	for i := range s.Tensors {
		e := &s.Tensors[i]
		if !bitnetProjections[e.Role] {
			continue
		}
		n := 1
		for d := 0; d < int(e.NDim); d++ {
			n *= int(e.Dims[d])
		}
		switch e.Type {
		case jlm.TypeTQ2:
		case jlm.TypeF32, jlm.TypeF16, jlm.TypeBF16:
			vals, err := floatsOf(e.Type, e.Data, n)
			if err != nil {
				return fmt.Errorf("convert: %s: %w", e.Name, err)
			}
			if e.Data, err = ternaryFromFloats(vals); err != nil {
				return fmt.Errorf("convert: %s: %w", e.Name, err)
			}
			e.Type = jlm.TypeTQ2
		default:
			return fmt.Errorf("convert: %s is %v: a BitNet projection is ternary, and its activations "+
				"are quantized per token for a ternary weight: %w", e.Name, e.Type, ErrNotImplemented)
		}
		sc, ok := scales[e.Name]
		if !ok {
			continue
		}
		used++
		if err := tq2Scale(e, sc); err != nil {
			return fmt.Errorf("convert: %s: %w", e.Name, err)
		}
	}
	if used != len(scales) {
		return fmt.Errorf("convert: %d projection scales and %d projections they scale", len(scales), used)
	}
	return nil
}

// tq2Scale multiplies every block's d of a TQ2_0 tensor by s, refusing a
// product that is not an f16.
func tq2Scale(e *jlm.Tensor, s float32) error {
	if len(e.Data)%66 != 0 {
		return fmt.Errorf("%d bytes is not whole TQ2_0 blocks", len(e.Data))
	}
	data := append([]byte(nil), e.Data...)
	for b := 0; b < len(data); b += 66 {
		d := float32(quant.DecodeHalf(binary.LittleEndian.Uint16(data[b+64:])))
		h, ok := exactHalf(d * s)
		if !ok {
			return fmt.Errorf("its scale %v times d %v is not an f16", s, d)
		}
		binary.LittleEndian.PutUint16(data[b+64:], h)
	}
	e.Data = data
	return nil
}

// elemsOf is a shape's element count.
func elemsOf(dims []uint64) uint64 {
	n := uint64(1)
	for _, d := range dims {
		n *= d
	}
	return n
}
