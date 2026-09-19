package laya

import (
	"fmt"
	"strings"
)

// Sequence is one tokenised question row: [CLS] <type> question: <instructions> [SEP]
// [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP], plus the position of every [MASK] marker.
type Sequence struct {
	IDs     []int64
	Markers []int
	QType   QType
}

// BuildSequence ports upstream laya.common.build_sequence (truncate_left=False, natural option order).
func BuildSequence(tok *Tokenizer, cfg *Config, stateText string, q *Question) (Sequence, error) {
	sp := cfg.SpecialTokens
	maskTok := sp.Mask
	opts := q.Options()

	ins := strings.ReplaceAll(q.Instructions, maskTok, " ")
	headIDs := tok.Encode(q.Type.String() + " question: " + ins)

	optIDs := make([][]int64, len(opts))
	for i, o := range opts {
		enc := tok.Encode(" " + strings.ReplaceAll(o, maskTok, " "))
		if len(enc) > 48 {
			enc = enc[:48]
		}
		row := make([]int64, 0, 1+len(enc))
		row = append(row, sp.MaskID)
		row = append(row, enc...)
		optIDs[i] = row
	}
	sumOpt := 0
	for _, o := range optIDs {
		sumOpt += len(o)
	}
	optBudget := cfg.HeadMaxLen - sumOpt
	if optBudget < 16 {
		per := max(4, (cfg.HeadMaxLen-16)/max(1, len(optIDs)))
		sumOpt = 0
		for i, o := range optIDs {
			if len(o) > per {
				optIDs[i] = o[:per]
			}
			sumOpt += len(optIDs[i])
		}
		optBudget = cfg.HeadMaxLen - sumOpt
	}
	if lim := max(8, optBudget); len(headIDs) > lim {
		headIDs = headIDs[:lim]
	}

	ids := make([]int64, 0, cfg.MaxLen)
	ids = append(ids, sp.ClsID)
	ids = append(ids, headIDs...)
	ids = append(ids, sp.SepID)
	markers := make([]int, 0, len(optIDs))
	for _, o := range optIDs {
		markers = append(markers, len(ids))
		ids = append(ids, o...)
	}
	ids = append(ids, sp.SepID)

	room := max(0, cfg.MaxLen-len(ids)-1)
	st := tok.Encode(strings.ReplaceAll(stateText, maskTok, " "))
	if len(st) > room {
		st = st[:room]
	}
	ids = append(ids, st...)
	ids = append(ids, sp.SepID)
	if len(ids) > cfg.MaxLen {
		ids = ids[:cfg.MaxLen]
	}
	kept := markers[:0]
	for _, m := range markers {
		if m < cfg.MaxLen {
			kept = append(kept, m)
		}
	}
	if len(kept) != len(opts) {
		return Sequence{}, fmt.Errorf("question %q options exceed head_max_len=%d", q.ID, cfg.HeadMaxLen)
	}
	return Sequence{IDs: ids, Markers: kept, QType: q.Type}, nil
}

// Batch is the padded tensor set fed to the ONNX graph (upstream collate_items).
type Batch struct {
	N, L, K       int
	InputIDs      []int64 // [N, L]
	AttentionMask []int64 // [N, L]
	MarkerPos     []int64 // [N, K]
	MarkerMask    []int64 // [N, K]
	QType         []int64 // [N]
	NumTokens     int     // sum of attention mask, reported as usage.input_tokens
}

// Collate pads sequences into one batch.
func Collate(seqs []Sequence, padID int64) *Batch {
	n := len(seqs)
	L, K := 0, 0
	for _, s := range seqs {
		L = max(L, len(s.IDs))
		K = max(K, len(s.Markers))
	}
	b := &Batch{N: n, L: L, K: K,
		InputIDs:      make([]int64, n*L),
		AttentionMask: make([]int64, n*L),
		MarkerPos:     make([]int64, n*K),
		MarkerMask:    make([]int64, n*K),
		QType:         make([]int64, n),
	}
	for i := range b.InputIDs {
		b.InputIDs[i] = padID
	}
	for i, s := range seqs {
		copy(b.InputIDs[i*L:], s.IDs)
		for j := range s.IDs {
			b.AttentionMask[i*L+j] = 1
		}
		b.NumTokens += len(s.IDs)
		for j, m := range s.Markers {
			b.MarkerPos[i*K+j] = int64(m)
			b.MarkerMask[i*K+j] = 1
		}
		b.QType[i] = int64(s.QType)
	}
	return b
}
