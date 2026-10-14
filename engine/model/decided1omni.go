package model

import (
	"fmt"
	"image"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// d1-omni's readout (jlm.DecisionD1Omni): one LFM2-encoder pass per question
// over
//
//	<bos> <state> state <q> instructions (<opt> <mask> option </opt>)* <decide>
//
// then Laya's head and scorer at each <mask>. The sequence is built as the
// reference's prompt.encode builds it: each piece tokenized on its own after
// `<|name|>` is rewritten to `<¦name¦>` (escape), the option block given
// max(96, min(24k+32, max_len/2)) tokens shared out evenly, the question cut
// to that block, the state to what is left.

// The reference's delimiters: reserved tokens of LFM2's vocabulary.
var d1OmniDelims = [...]string{"<|reserved_7|>", "<|reserved_8|>", "<|reserved_9|>", "<|reserved_10|>",
	"<|reserved_11|>", "<|mask|>"}

// d1OmniMaxLen is the text's room: the reference's max_length, the
// container's context, less a picture prefix and cut to image_text_length
// when there is one.
func (d *Decider) d1OmniMaxLen() int {
	p := len(d.omniPre) / d.m.Cfg.NEmbd
	if p == 0 {
		return d.m.Cfg.NCtx
	}
	return min(d1OmniImageText, d.m.Cfg.NCtx-p)
}

// d1OmniSequence is one question's ids and the position of each option's
// marker, in the order the model reads them (a noul false first).
func (d *Decider) d1OmniSequence(state jinja.Value, q *DecisionQuestion, maxLen int) ([]int32, []int, error) {
	v := d.m.Vocab
	var dl [len(d1OmniDelims)]int32
	for i, s := range d1OmniDelims {
		id, ok := v.ID(s)
		if !ok {
			return nil, nil, fmt.Errorf("the vocabulary has no %s", s)
		}
		dl[i] = id
	}
	st, qd, od, oe, dec, mask := dl[0], dl[1], dl[2], dl[3], dl[4], dl[5]
	bos := d.m.container.Vocab().BOS
	if bos < 0 {
		return nil, nil, fmt.Errorf("the vocabulary has no BOS")
	}
	enc := func(s string) []int32 { return v.EncodeSpecial(d1OmniEscape(s), false) }
	opts := d1OmniOptions(q, len(d.omniPre) > 0)
	k := len(opts)
	budget := max(96, min(k*24+32, maxLen/2))
	per := max(2, (budget-3*k)/k)
	question := append([]int32{qd}, enc(pyStr(q.Instructions, false))...)
	question = trim(question, max(16, budget))
	markers := make([]int, k)
	for i, o := range opts {
		markers[i] = len(question) + 1
		question = append(question, od, mask)
		question = append(question, trim(enc(" "+o), per)...)
		question = append(question, oe)
	}
	question = append(question, dec)
	room := max(0, maxLen-len(question)-2)
	stIDs := append([]int32{st}, trim(enc(d1OmniState(state)), room)...)
	ids := append([]int32{bos}, stIDs...)
	ids = trim(append(ids, question...), maxLen)
	for i := range markers {
		markers[i] += 1 + len(stIDs)
	}
	if markers[k-1] >= maxLen {
		return nil, nil, fmt.Errorf("the options do not fit in the context")
	}
	return ids, markers, nil
}

// d1OmniEscape is prompt.escape: `<|name|>` (name of [A-Za-z0-9_]) becomes
// `<¦name¦>`, so caller text cannot emit a delimiter or a marker.
func d1OmniEscape(s string) string {
	if !strings.Contains(s, "<|") {
		return s
	}
	var b strings.Builder
	for {
		i := strings.Index(s, "<|")
		if i < 0 {
			b.WriteString(s)
			return b.String()
		}
		b.WriteString(s[:i])
		j := i + 2
		for j < len(s) && (s[j] == '_' || s[j] >= '0' && s[j] <= '9' || s[j] >= 'A' && s[j] <= 'Z' || s[j] >= 'a' && s[j] <= 'z') {
			j++
		}
		if j > i+2 && strings.HasPrefix(s[j:], "|>") {
			b.WriteString("<¦" + s[i+2:j] + "¦>")
			s = s[j+2:]
			continue
		}
		b.WriteString("<|")
		s = s[i+2:]
	}
}

// d1OmniState is prompt.serialize: a string as it is, anything else
// json.dumps(ensure_ascii=False), insertion order.
func d1OmniState(state jinja.Value) string {
	if state.IsUndefined() || state.IsNone() {
		return ""
	}
	if state.IsString() {
		return state.AsString()
	}
	var b strings.Builder
	pyJSON(&b, state, ", ", ": ", false)
	return b.String()
}

// d1OmniCriterion is prompt._criterion: a string as it is, anything else
// json.dumps(ensure_ascii=False, separators=(", ", ": ")).
func d1OmniCriterion(v jinja.Value) string {
	if v.IsString() {
		return v.AsString()
	}
	var b strings.Builder
	pyJSON(&b, v, ", ", ": ", false)
	return b.String()
}

// d1OmniOptions is prompt.render_options: a noul as [false, true], with its
// default wording where the request gives none -- after pictures the
// reference's YES_NO, "no" and "yes".
func d1OmniOptions(q *DecisionQuestion, yesNo bool) []string {
	out := make([]string, len(q.Options))
	for i, o := range q.Options {
		desc := o.Description
		empty := desc.IsUndefined() || desc.IsNone() || (desc.IsString() && desc.AsString() == "")
		switch {
		case q.Type == jlm.QuestionChoice && empty:
			out[i] = o.Key
		case q.Type == jlm.QuestionChoice:
			out[i] = o.Key + ": " + d1OmniCriterion(desc)
		case q.Type == jlm.QuestionScore:
			out[i] = "level " + strconv.Itoa(i) + ": " + d1OmniCriterion(desc)
		case empty && yesNo && o.Key == "true":
			out[i] = "true: yes"
		case empty && yesNo:
			out[i] = "false: no"
		case empty && o.Key == "true":
			out[i] = "true: yes, the statement holds"
		case empty:
			out[i] = "false: no, the statement does not hold"
		default:
			out[i] = o.Key + ": " + d1OmniCriterion(desc)
		}
	}
	return out
}

// d1OmniForward encodes ids, runs the head at the markers and returns the
// scores.
func (d *Decider) d1OmniForward(ids []int32, qt jlm.QuestionType, markers []int, v int) ([]float32, error) {
	if err := d.emb.lfm2Encode(ids, d.omniPre); err != nil {
		return nil, err
	}
	out := d.variantScores(v, len(markers))
	if err := d.emb.layaScores(out, len(ids), qt, markers); err != nil {
		return nil, err
	}
	d.input += len(ids) + len(d.omniPre)/d.m.Cfg.NEmbd
	return out, nil
}

// d1-omni's pictures (vision.py): each picture cut as LFM2-VL cuts it (tiles
// of 512 pixels and a thumbnail, resampled bilinear), every piece through the
// tower and projector, and the rows of every piece of every picture, in
// order, a prefix in front of the prompt's embeddings. No marker tokens: the
// prefix's rows are told apart by the mask (a prefix row attends to the
// prefix alone) and the convolution's cut at the prefix's end.

// d1OmniImageText is the reference's image_text_length: a prompt after
// pictures is cut to it (D1OmniConfig's default; the GGUF does not carry it).
const d1OmniImageText = 896

// d1OmniPrefix encodes the request's pictures into d.omniPre, the prefix
// rows the prompt's embeddings follow, on the Decider's vision segment.
func (d *Decider) d1OmniPrefix(imgs []image.Image) error {
	tw := d.m.Tower()
	if d.omniVis == nil {
		d.omniVis = tw.newState(d.emb.jit, false)
	}
	d.omniPre = d.omniPre[:0]
	for i, img := range imgs {
		if img == nil {
			return fmt.Errorf("images[%d] is empty", i)
		}
		_, _, pieces, err := tw.LFM2Pieces(img)
		if err != nil {
			return fmt.Errorf("images[%d]: %w", i, err)
		}
		for _, pc := range pieces {
			rows, err := d.omniVis.encodePicture(pc)
			if err != nil {
				return fmt.Errorf("images[%d]: %w", i, err)
			}
			d.omniPre = append(d.omniPre, rows...)
		}
	}
	if d.d1OmniMaxLen() < 64 {
		return fmt.Errorf("the images take %d of the %d positions; send fewer images",
			len(d.omniPre)/d.m.Cfg.NEmbd, d.m.Cfg.NCtx)
	}
	return nil
}
