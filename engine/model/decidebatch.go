package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Decisions in a batch: several requests answered at once, as a server
// answers the requests that arrived while it was busy with one. A decoder
// readout's prompts -- every question and every variant of every request --
// run as rows of one batch session (Scheduler), a step reading the weights
// once for every prompt in it, and each prompt's label logits are read where
// its last row comes out. An encoder readout (Laya) runs its sequences one
// encoder pass each: its blocks are non-causal over the call's rows, so
// sequences packed into one call would attend to each other.

// DecisionRequest is one request's state and questions.
type DecisionRequest struct {
	State     jinja.Value
	Questions []DecisionQuestion
}

// decisionBatchRows is the most prompts one batch session runs at once.
const decisionBatchRows = 16

// DecisionResult is one request's part of a batch: its answers and the
// tokens its prompts ran, or why it failed alone.
type DecisionResult struct {
	Answers     []DecisionAnswer
	InputTokens int
	Err         error
}

// DecideBatch answers every request, in order: out[i] is what Decide would
// answer for reqs[i] alone. A request that fails its checks fails alone
// (out[i].Err); an error running the batch fails it whole.
func (d *Decider) DecideBatch(reqs []DecisionRequest) ([]DecisionResult, error) {
	out := make([]DecisionResult, len(reqs))
	type job struct {
		r, q, v int
		ids     []int32
		groups  [][]int32
		markers []int
		scores  []float32
	}
	var jobs []*job
	for r := range reqs {
		rq, res := &reqs[r], &out[r]
		if res.Err = d.Prepare(rq.State, rq.Questions); res.Err != nil {
			continue
		}
		var mine []*job
	questions:
		for qi := range rq.Questions {
			q := &rq.Questions[qi]
			for v := range d.variants(q) {
				j := &job{r: r, q: qi, v: v}
				var err error
				if d.kind == jlm.DecisionLaya {
					j.ids, j.markers, err = d.layaSequence(rq.State, q)
				} else {
					j.ids, j.groups, err = d.prompt(rq.State, q, v)
				}
				if err != nil {
					res.Err = fmt.Errorf("questions.%s: %w", q.ID, err)
					break questions
				}
				mine = append(mine, j)
			}
		}
		if res.Err == nil {
			jobs = append(jobs, mine...)
		}
	}
	d.input = 0
	for _, j := range jobs {
		out[j.r].InputTokens += len(j.ids)
		d.input += len(j.ids)
	}
	if d.kind == jlm.DecisionLaya {
		for _, j := range jobs {
			s, err := d.layaForward(j.ids, reqs[j.r].Questions[j.q].Type, j.markers, 0)
			if err != nil {
				return nil, err
			}
			j.scores = append([]float32(nil), s...)
		}
	} else if len(jobs) > 0 {
		sc, err := d.batchSession(min(len(jobs), decisionBatchRows))
		if err != nil {
			return nil, err
		}
		var failed error
		for _, j := range jobs {
			sc.Submit(&Seq{Prompt: j.ids, MaxTokens: 1, Sample: func(lg []float32) int32 {
				s, err := d.labelScores(lg, j.groups)
				if err != nil && failed == nil {
					failed = err
				}
				j.scores = s
				return 0
			}})
		}
		for sc.Pending() > 0 || sc.Live() > 0 {
			done, err := sc.Step()
			if err != nil {
				return nil, err
			}
			for _, s := range done {
				if s.Err != nil {
					return nil, s.Err
				}
			}
		}
		if failed != nil {
			return nil, failed
		}
	}
	// The answers, each from its question's variants, in the request's order.
	at := 0
	for r := range reqs {
		res := &out[r]
		if res.Err != nil {
			continue
		}
		qs := reqs[r].Questions
		res.Answers = make([]DecisionAnswer, len(qs))
		for qi := range qs {
			var all [][]float32
			for ; at < len(jobs) && jobs[at].r == r && jobs[at].q == qi; at++ {
				all = append(all, jobs[at].scores)
			}
			a, err := d.answer(&qs[qi], all)
			if err != nil {
				res.Answers, res.Err = nil, fmt.Errorf("questions.%s: %w", qs[qi].ID, err)
				break
			}
			res.Answers[qi] = a
		}
		for at < len(jobs) && jobs[at].r == r {
			at++
		}
	}
	return out, nil
}

// batchSession is the Decider's batch session, at least rows wide, placed on
// the Decider's device as its own State is. It is kept between batches; a
// wider batch replaces it.
func (d *Decider) batchSession(rows int) (*Scheduler, error) {
	if d.batch != nil && len(d.batch.live) >= rows {
		return d.batch, nil
	}
	if d.batch != nil {
		d.batch.Close()
		d.batch = nil
	}
	sc := NewScheduler(d.m, rows, d.st.MaxSeq())
	if d.dev != nil {
		if err := sc.st.SetDeviceLayers(d.dev, d.devMax); err != nil {
			sc.Close()
			return nil, err
		}
	}
	d.batch = sc
	return sc, nil
}

// BatchDeviceBlocks is how many blocks the Decider's batch session runs on a
// device: 0 before a batch has run.
func (d *Decider) BatchDeviceBlocks() int {
	if d.batch == nil {
		return 0
	}
	return d.batch.st.GPULayers()
}

// BatchStats is what the Decider's batch session has run.
func (d *Decider) BatchStats() SchedulerStats {
	if d.batch == nil {
		return SchedulerStats{}
	}
	return d.batch.Stats()
}

// setBatchDevice follows the Decider's device with its batch session.
func (d *Decider) setBatchDevice(dev nn.Device, max int) error {
	d.dev, d.devMax = dev, max
	if d.batch == nil {
		return nil
	}
	return d.batch.st.SetDeviceLayers(dev, max)
}
