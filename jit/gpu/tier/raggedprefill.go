package tier

import (
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A ragged step's prompt chunks attend as a prefill
// (docs/design/ragged-prefill.md). A step's rows are several sequences', and
// a prompt chunk among them used to attend through the decode plan a row at a
// time, every row reading its sequence's whole history on its own. Here each
// sequence with more than one row in the step gets its rows into a run
// table, in position order and padded to whole workgroups with its last row,
// and one launch of the flash prefill's run-table form (kernels'
// FlashPrefill70Shape.Map) attends them all, q and the output at the step's
// own rows. The decoding rows -- one row of their sequence -- keep the decode
// plan, run first over descriptors in which the chunk rows are empty, and the
// table's launch then writes the chunk rows over whatever that left.
//
// The table is data the launch reads (pMap), staged with the descriptors, so
// a recording serves every step of the same padded width (graphKey.ragMap).

// ragTable is a ragged call's run table: the step rows of each sequence with
// more than one row, padded per workgroup. Its slot count is a multiple of wg.
// alone reports that every real row is in it -- no sequence has a single
// row, which the decode plan would take -- and then the call's padding rows
// are in it too, a run of their own whose descriptors attend nothing, so the
// table's launch writes every row and the decode plan need not run. It is
// built into dst and returned; dst is reused, so a step allocates nothing
// once the widest table has been made.
func ragTable(dst []uint32, rows []pagedRow, wg int, order []int) (_ []uint32, _ []int, alone bool) {
	dst = dst[:0]
	order = order[:0]
	for r, row := range rows {
		if !row.pad {
			order = append(order, r)
		}
	}
	// By sequence, then position: each sequence's rows a contiguous run.
	slices.SortFunc(order, func(a, b int) int {
		ra, rb := rows[a], rows[b]
		if ra.s != rb.s {
			if ra.s.sid != rb.s.sid {
				if ra.s.sid < rb.s.sid {
					return -1
				}
				return 1
			}
			return ra.s.base - rb.s.base
		}
		return ra.pos - rb.pos
	})
	alone = true
	for i := 0; i < len(order); {
		j := i + 1
		for j < len(order) && rows[order[j]].s == rows[order[i]].s {
			j++
		}
		if j-i > 1 {
			for k := 0; k < (j-i+wg-1)/wg*wg; k++ {
				dst = append(dst, uint32(order[min(i+k, j-1)]))
			}
		} else {
			alone = false
		}
		i = j
	}
	if alone && len(dst) > 0 {
		first, n := len(dst), 0
		for r, row := range rows {
			if row.pad {
				dst = append(dst, uint32(r))
				n++
			}
		}
		for n > 0 && (len(dst)-first)%wg != 0 {
			dst = append(dst, dst[len(dst)-1])
		}
	}
	return dst, order, alone
}

// ragPrep readies bs's ragged call for the run-table launch, or reports false
// (the decode plan takes every row) where the call has no run of two rows or
// the device has no run-table form: the flash prefill must be one with a
// table (FlashPrefill80 and its m16n8k8 twin, FlashPrefillTile), with no
// sinks and no streamed history. It builds the table and the decode descriptors with
// the table's rows emptied, compiles the launch at the table's width, and
// grows the table's buffer. Callers hold g.mu and have run pagedStage.
func (g *devTier) ragPrep(bs *blockScratch) bool {
	pk := bs.pkv
	pk.ragRows = 0
	pf := pk.pf
	if pf == nil || pf.ragBuild == nil || pf.fs.Splits != 0 || pk.st != nil || g.NoRagPrefill {
		return false
	}
	wg := pf.ragWG
	pk.ragMap, pk.ragOrder, pk.ragAlone = ragTable(pk.ragMap, pk.rows, wg, pk.ragOrder)
	if len(pk.ragMap) == 0 {
		return false
	}
	n := len(pk.ragMap)
	if pf.ragK == nil {
		pf.ragK = map[int]backend.Kernel{}
	}
	if pf.ragK[n] == nil {
		fs := pf.fs
		fs.Map, fs.Rows = true, n
		k, err := pf.ragBuild(fs)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		c, err := g.compileShared(k)
		if err != nil {
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		pf.ragK[n] = c
	}
	if pk.pMapBytes < 4*n {
		if pk.pMap != nil {
			pk.pMap.Free()
		}
		b, err := g.allocBytes(make([]byte, 4*n))
		if err != nil {
			pk.pMap, pk.pMapBytes = nil, 0
			g.LastErr = "ragged prefill: " + err.Error()
			return false
		}
		pk.pMap, pk.pMapBytes = b, 4*n
	}
	for _, dec := range []struct {
		buf  *backend.Buf
		src  []uint32
		dst  *[]uint32
		size *int
	}{{&pk.pRowDec, pk.desc, &pk.descDec, &pk.pRowDecBytes}, {&pk.pRowDecW, pk.descW, &pk.descDecW, &pk.pRowDecWBytes}} {
		if len(dec.src) == 0 {
			*dec.dst = (*dec.dst)[:0]
			continue
		}
		d := append((*dec.dst)[:0], dec.src...)
		// Empty at the row's own end: the decode plan attends nothing there,
		// and the table's launch writes the row.
		for _, r := range pk.ragMap {
			w := int(r) * kernels.PRowWords
			d[w+kernels.PRowStart] = d[w+kernels.PRowEnd]
		}
		*dec.dst = d
		if *dec.size < 4*len(d) {
			if *dec.buf != nil {
				(*dec.buf).Free()
			}
			b, err := g.allocBytes(make([]byte, 4*len(d)))
			if err != nil {
				*dec.buf, *dec.size = nil, 0
				g.LastErr = "ragged prefill: " + err.Error()
				return false
			}
			*dec.buf, *dec.size = b, 4*len(d)
		}
	}
	pk.ragRows = n
	return true
}

// ragWrites stages the run table and the decode descriptors (ragPrep's).
func (pk *pagedScratch) ragWrites(w func(backend.Buf, []byte)) {
	if pk.ragRows == 0 {
		return
	}
	w(pk.pMap, u32view(pk.ragMap))
	w(pk.pRowDec, u32view(pk.descDec))
	if len(pk.descDecW) > 0 {
		w(pk.pRowDecW, u32view(pk.descDecW))
	}
}

// ragAttend launches the run table's flash prefill over the chunk rows,
// through desc (the full or the windowed set).
func (pk *pagedScratch) ragAttend(lc *launcher, q, k, v, n, out, tab, desc backend.Buf) error {
	fs := pk.pf.fs
	fs.Map, fs.Rows = true, pk.ragRows
	return lc.launch(pk.pf.ragK[pk.ragRows], pk.pf.ragGroups(fs), pk.pf.ragThreads,
		q, k, v, n, out, tab, desc, pk.pMap)
}

// ragFree gives back the run table's buffers. They are a few words a row,
// allocated as the recurrent seats' zeros are, outside the budget.
func (pk *pagedScratch) ragFree() {
	for _, b := range []*backend.Buf{&pk.pMap, &pk.pRowDec, &pk.pRowDecW} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	pk.pMapBytes, pk.pRowDecBytes, pk.pRowDecWBytes = 0, 0, 0
}
