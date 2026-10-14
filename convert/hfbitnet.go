package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// bitnetHF is BitNetForCausalLM from safetensors: llama's names with the two
// inner norms, every projection ternary. transformers runs it through its
// "bitnet" quantizer, whose quantization_config says which form the
// projections are stored in (integrations/bitnet.py):
//
//	autobitlinear, offline   uint8, four 2-bit codes a byte (code-1 is the
//	                         weight), and a bf16 weight_scale the output is
//	                         multiplied by -- the published release
//	autobitlinear, online    the latent weights, ternarized every call by
//	                         WeightQuant: s = 1/mean|W|, round(W*s) clamped to
//	                         [-1, 1], over s
//	bitlinear                the uint8 codes with a weight_scale the output is
//	                         DIVIDED by
//
// Each becomes TypeTQ2 (convert/bitnet.go), with the weight's magnitude as
// every block's d where that is an f16 exactly, and refused where it is not.
var bitnetHF = &hfArch{
	arch:        jlm.ArchBitNet,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block: blockPlus(hfLlamaBlockNoBias, map[string]jlm.Role{
		"self_attn.attn_sub_norm.weight": jlm.RoleAttnSubNorm,
		"mlp.ffn_sub_norm.weight":        jlm.RoleFFNSubNorm,
	}),
	ignore: llamaHF.ignore,
	config: bitnetHFConfig,
	gather: bitnetGather,
	retype: bitnetRetype,
}

// bitnetForm is how a checkpoint stores its projections.
type bitnetForm int

const (
	bitnetPackedMul bitnetForm = iota + 1 // autobitlinear, offline
	bitnetLatent                          // autobitlinear, online
	bitnetPackedDiv                       // bitlinear
)

// bitnetQuant is the "bitnet" quantization_config.
type bitnetQuant struct {
	QuantMethod string          `json:"quant_method"`
	LinearClass string          `json:"linear_class"`
	Mode        string          `json:"quantization_mode"`
	UseRMSNorm  *bool           `json:"use_rms_norm"`
	NotConvert  json.RawMessage `json:"modules_to_not_convert"`
}

// bitnetFormOf reads the quantization_config. A BitNetForCausalLM without
// one runs full-precision linears in transformers, which is not the ternary
// graph this arch is, so it is refused; so is every option that changes the
// graph (an RMSNorm inside each linear, another module list).
func bitnetFormOf(c *hfConfig) (bitnetForm, error) {
	raw, ok := c.keys["quantization_config"]
	if !ok || empty(raw) {
		return 0, fmt.Errorf("convert: %s: a BitNetForCausalLM with no quantization_config runs "+
			"full-precision projections in transformers: %w", c.path, ErrNotImplemented)
	}
	var q bitnetQuant
	if err := json.Unmarshal(raw, &q); err != nil {
		return 0, fmt.Errorf("convert: %s: quantization_config: %w", c.path, err)
	}
	if q.QuantMethod != "bitnet" {
		return 0, fmt.Errorf("convert: %s: quant_method %q is not BitNet's: %w", c.path, q.QuantMethod, ErrNotImplemented)
	}
	if q.UseRMSNorm != nil && *q.UseRMSNorm {
		return 0, fmt.Errorf("convert: %s: use_rms_norm puts an RMSNorm inside every projection: %w",
			c.path, ErrNotImplemented)
	}
	if !empty(q.NotConvert) {
		var mods []string
		if err := json.Unmarshal(q.NotConvert, &mods); err != nil {
			return 0, fmt.Errorf("convert: %s: modules_to_not_convert: %w", c.path, err)
		}
		for _, m := range mods {
			if m != "lm_head" {
				return 0, fmt.Errorf("convert: %s: modules_to_not_convert names %q, a projection "+
					"left full precision: %w", c.path, m, ErrNotImplemented)
			}
		}
	}
	class := q.LinearClass
	if class == "" {
		class = "bitlinear" // BitNetQuantConfig's default
	}
	mode := q.Mode
	if mode == "" {
		mode = "offline"
	}
	switch {
	case class == "autobitlinear" && mode == "offline":
		return bitnetPackedMul, nil
	case class == "autobitlinear" && mode == "online":
		return bitnetLatent, nil
	case class == "bitlinear" && mode == "offline":
		return bitnetPackedDiv, nil
	}
	return 0, fmt.Errorf("convert: %s: linear_class %q in mode %q: %w", c.path, class, mode, ErrNotImplemented)
}

// bitnetHFConfig reads BitNetConfig with its class defaults, and the
// quantization_config's form.
func bitnetHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	if _, err := bitnetFormOf(c); err != nil {
		return nil, err
	}
	hfClassDefaults{kvHeads: 5, ropeTheta: 500000, rmsEps: 1e-5, tied: false, nCtx: 2048}.apply(c)
	if c.act() == "" {
		c.HiddenAct = "relu2"
	}
	cfg, err := hfConfigWith(jlm.ArchBitNet, 0, "relu2")(c, names)
	if err != nil {
		return nil, err
	}
	bitnetConfig(cfg)
	return cfg, nil
}

// bitnetGather consumes every projection's weight_scale: it is read with its
// weight (bitnetRetype), never stored. A weight_scale with no projection, or
// a form that has none, is refused.
func bitnetGather(c *jlm.Config, names []string, _ func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
	used := map[string]bool{}
	have := map[string]bool{}
	for _, n := range names {
		have[n] = true
	}
	for _, n := range names {
		if !strings.HasSuffix(n, ".weight_scale") {
			continue
		}
		w := strings.TrimSuffix(n, "_scale")
		if !have[w] {
			return nil, nil, fmt.Errorf("convert: %s scales no weight", n)
		}
		used[n] = true
	}
	return nil, used, nil
}

// bitnetRetype gives a projection its ternary type before anything is laid
// out, and the read that produces its TQ2_0 bytes.
func bitnetRetype(hc *hfConfig, role jlm.Role, name string, t *safetensors.Tensor,
	read func(name string) ([]byte, *safetensors.Tensor, error)) (*hfRetyped, error) {
	if !bitnetProjections[role] {
		return nil, nil
	}
	form, err := bitnetFormOf(hc)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("convert: %s has %d dimensions", name, len(t.Shape))
	}
	switch form {
	case bitnetLatent:
		if t.DType != safetensors.BF16 {
			return nil, fmt.Errorf("convert: %s is %s; BitNet's latent weights are ternarized as the "+
				"class runs them, in bf16: %w", name, t.DType, ErrNotImplemented)
		}
		out, in := t.Shape[0], t.Shape[1]
		return &hfRetyped{ty: jlm.TypeTQ2, shape: []uint64{out, in}, read: func() ([]byte, error) {
			b, _, err := read(name)
			if err != nil {
				return nil, err
			}
			tq, err := bitnetTernarizeBF16(b)
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			return tq, nil
		}}, nil
	default:
		if t.DType != safetensors.U8 {
			return nil, fmt.Errorf("convert: %s is %s, and this checkpoint's projections are "+
				"packed codes (U8)", name, t.DType)
		}
		out, in := 4*t.Shape[0], t.Shape[1]
		return &hfRetyped{ty: jlm.TypeTQ2, shape: []uint64{out, in}, read: func() ([]byte, error) {
			b, _, err := read(name)
			if err != nil {
				return nil, err
			}
			sb, st, err := read(name + "_scale")
			if err != nil {
				return nil, err
			}
			ws, err := bitnetScalar(st, sb)
			if err != nil {
				return nil, fmt.Errorf("convert: %s_scale: %w", name, err)
			}
			s := ws
			if form == bitnetPackedDiv {
				s = 1 / ws
			}
			d, ok := exactHalf(s)
			if !ok {
				return nil, fmt.Errorf("convert: %s: its magnitude %v is not an f16: %w", name, s, ErrNotImplemented)
			}
			tq, err := bitnetUnpack(b, int(out), int(in), d)
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			return tq, nil
		}}, nil
	}
}

// bitnetScalar reads a one-element float tensor.
func bitnetScalar(t *safetensors.Tensor, b []byte) (float32, error) {
	n := uint64(1)
	for _, d := range t.Shape {
		n *= d
	}
	if n != 1 {
		return 0, fmt.Errorf("%d values, want one", n)
	}
	switch t.DType {
	case safetensors.BF16:
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16), nil
	case safetensors.F32:
		return math.Float32frombits(binary.LittleEndian.Uint32(b)), nil
	case safetensors.F16:
		return float32(quant.DecodeHalf(binary.LittleEndian.Uint16(b))), nil
	}
	return 0, fmt.Errorf("dtype %s", t.DType)
}

// bitnetUnpack turns transformers' packed codes into TQ2_0 blocks: a [out/4,
// in] uint8 tensor whose byte (r, i) holds row j*out/4 + r's code for column
// i in bits 2j (pack_weights / unpack_weights), the code being the weight
// plus one.
func bitnetUnpack(b []byte, out, in int, d uint16) ([]byte, error) {
	rows4 := out / 4
	if out%4 != 0 || len(b) != rows4*in || in%256 != 0 {
		return nil, fmt.Errorf("%d bytes for a [%d, %d] ternary weight", len(b), out, in)
	}
	dst := make([]byte, out*in/256*66)
	var bad error
	for o := 0; o < out; o++ {
		src := b[(o%rows4)*in : (o%rows4)*in+in]
		sh := uint(2 * (o / rows4))
		for blk := 0; blk < in/256; blk++ {
			tq2Put(dst[(o*in/256+blk)*66:], func(e int) uint8 {
				c := src[blk*256+e] >> sh & 3
				if c == 3 && bad == nil {
					bad = fmt.Errorf("row %d column %d holds code 3, which is no ternary weight", o, blk*256+e)
				}
				return c
			}, d)
		}
	}
	return dst, bad
}

// bitnetTernarizeBF16 is WeightQuant over a bf16 [out, in] weight, step by
// step as torch runs it on bf16 tensors (each op computed in f32 and rounded
// to bf16, round to nearest even):
//
//	m = bf16(sum|w| / n)          the mean, accumulated wide
//	s = bf16(1 / m)               after clamp(min=1e-5)
//	q = clamp(rne(bf16(w*s)), -1, 1)
//	weight = bf16(q / s)          so the magnitude is bf16(1/s)
//
// The magnitude is the blocks' d, which a bf16 in f16's normal range is
// exactly.
func bitnetTernarizeBF16(b []byte) ([]byte, error) {
	n := len(b) / 2
	if n%256 != 0 {
		return nil, fmt.Errorf("%d values is not whole 256-blocks", n)
	}
	w := make([]float32, n)
	sum := 0.0
	for i := range w {
		w[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
		sum += math.Abs(float64(w[i]))
	}
	m := bf16Round(float32(sum / float64(n)))
	if m < bf16Round(1e-5) {
		m = bf16Round(1e-5)
	}
	s := bf16Round(1 / m)
	mag := bf16Round(1 / s)
	d, ok := exactHalf(mag)
	if !ok {
		return nil, fmt.Errorf("its magnitude %v is not an f16", mag)
	}
	out := make([]byte, n/256*66)
	for blk := 0; blk < n/256; blk++ {
		v := w[blk*256 : blk*256+256]
		tq2Put(out[blk*66:], func(e int) uint8 {
			q := math.RoundToEven(float64(bf16Round(v[e] * s)))
			return uint8(max(-1, min(1, q)) + 1)
		}, d)
	}
	return out, nil
}

// bf16Round is x rounded to the nearest bfloat16, ties to even, as torch's
// float-to-bfloat16 conversion rounds (NaN is kept a NaN).
func bf16Round(x float32) float32 {
	u := math.Float32bits(x)
	if x != x {
		return x
	}
	u += 0x7FFF + (u >> 16 & 1)
	return math.Float32frombits(u &^ 0xFFFF)
}
