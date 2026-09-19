package laya

import (
	"fmt"
	"strconv"
)

// QType is the typed-question kind, numbered as in the upstream QTYPES table.
type QType int

const (
	QChoice QType = 0
	QScore  QType = 1
	QNoul   QType = 2
)

func (t QType) String() string {
	switch t {
	case QChoice:
		return "choice"
	case QScore:
		return "score"
	case QNoul:
		return "noul"
	}
	return "?"
}

// Question is the normalised internal form of one typed question (upstream Agent._to_internal).
type Question struct {
	ID           string
	Type         QType
	Instructions string   // always text; structured instructions are JSON-dumped
	Keys         []string // choice: option labels in order
	Descs        []Value  // choice: description per key (null = none)
	Levels       []Value  // score: rubric levels in order
	NoulFalse    Value    // noul: optional description of the false side
	NoulTrue     Value    // noul: optional description of the true side
}

// Spec is an ordered list of questions, the API's "spec" object.
type Spec []Question

// ParseSpec validates and normalises a spec object: {"qid": {"type": ..., "instructions": ..., "criteria": ...}, ...}.
func ParseSpec(v Value) (Spec, error) {
	if v.Kind != KindObject {
		return nil, fmt.Errorf("spec must be a JSON object mapping question ids to definitions, got %s", v.Kind)
	}
	if len(v.Obj) == 0 {
		return nil, fmt.Errorf("spec has no questions")
	}
	out := make(Spec, 0, len(v.Obj))
	for _, f := range v.Obj {
		q, err := ParseQuestion(f.Key, f.Val)
		if err != nil {
			return nil, err
		}
		out = append(out, q)
	}
	return out, nil
}

// ParseQuestion normalises a single question definition.
func ParseQuestion(id string, def Value) (Question, error) {
	q := Question{ID: id}
	if def.Kind != KindObject {
		return q, fmt.Errorf("question %q: definition must be an object", id)
	}
	tv, ok := def.Get("type")
	if !ok || tv.Kind != KindString {
		return q, fmt.Errorf("question %q: missing string field \"type\" (choice|score|noul)", id)
	}
	switch tv.Str {
	case "choice":
		q.Type = QChoice
	case "score":
		q.Type = QScore
	case "noul":
		q.Type = QNoul
	default:
		return q, fmt.Errorf("question %q: unknown type %q (choice|score|noul)", id, tv.Str)
	}

	ins, ok := def.Get("instructions")
	if !ok {
		return q, fmt.Errorf("question %q: missing field \"instructions\"", id)
	}
	if ins.Kind == KindString {
		q.Instructions = ins.Str
	} else {
		q.Instructions = PyDumps(ins, DumpOptions{EnsureASCII: true})
	}

	crit, hasCrit := def.Get("criteria")
	switch q.Type {
	case QChoice:
		if !hasCrit || crit.IsNull() {
			return q, fmt.Errorf("question %q: choice questions need \"criteria\" (object or array of labels)", id)
		}
		switch crit.Kind {
		case KindObject:
			for _, f := range crit.Obj {
				q.Keys = append(q.Keys, f.Key)
				q.Descs = append(q.Descs, f.Val)
			}
		case KindArray:
			seen := map[string]bool{}
			for _, e := range crit.Arr {
				if e.Kind != KindString {
					return q, fmt.Errorf("question %q: criteria list items must be strings", id)
				}
				k := e.Str
				if seen[k] {
					continue
				}
				seen[k] = true
				q.Keys = append(q.Keys, k)
				q.Descs = append(q.Descs, NullValue)
			}
		default:
			return q, fmt.Errorf("question %q: criteria must be an object or array, got %s", id, crit.Kind)
		}
		if len(q.Keys) == 0 {
			return q, fmt.Errorf("question %q: choice questions need at least one option", id)
		}
	case QScore:
		if !hasCrit || crit.Kind != KindArray {
			return q, fmt.Errorf("question %q: score questions need \"criteria\" as an array of rubric levels", id)
		}
		if len(crit.Arr) < 2 {
			return q, fmt.Errorf("question %q: score questions need at least two rubric levels", id)
		}
		q.Levels = crit.Arr
	case QNoul:
		q.NoulFalse, q.NoulTrue = NullValue, NullValue
		if hasCrit && crit.Kind == KindObject {
			if fv, ok := crit.Get("false"); ok {
				q.NoulFalse = fv
			}
			if tv, ok := crit.Get("true"); ok {
				q.NoulTrue = tv
			}
		}
	}
	return q, nil
}

// renderCriterion mirrors upstream render_criterion: strings pass through, anything else is
// compact JSON (ensure_ascii=False, separators=(", ", ": ")).
func renderCriterion(v Value) string {
	if v.Kind == KindString {
		return v.Str
	}
	return PyDumps(v, DumpOptions{EnsureASCII: false})
}

// isEmptyCriterion reports the upstream test `v is None or v == ""`.
func isEmptyCriterion(v Value) bool {
	return v.Kind == KindNull || (v.Kind == KindString && v.Str == "")
}

// Options renders the option texts in label-index order (upstream render_options).
// Noul is always [false, true].
func (q *Question) Options() []string {
	switch q.Type {
	case QChoice:
		out := make([]string, len(q.Keys))
		for i, k := range q.Keys {
			if isEmptyCriterion(q.Descs[i]) {
				out[i] = k
			} else {
				out[i] = k + ": " + renderCriterion(q.Descs[i])
			}
		}
		return out
	case QScore:
		out := make([]string, len(q.Levels))
		for i, c := range q.Levels {
			out[i] = "level " + strconv.Itoa(i) + ": " + renderCriterion(c)
		}
		return out
	default:
		f := "no, the statement does not hold"
		if !isEmptyCriterion(q.NoulFalse) {
			f = renderCriterion(q.NoulFalse)
		}
		t := "yes, the statement holds"
		if !isEmptyCriterion(q.NoulTrue) {
			t = renderCriterion(q.NoulTrue)
		}
		return []string{"false: " + f, "true: " + t}
	}
}

// NumOptions is the number of options k for temperature bucketing and confidence.
func (q *Question) NumOptions() int {
	switch q.Type {
	case QChoice:
		return len(q.Keys)
	case QScore:
		return len(q.Levels)
	}
	return 2
}

// SerializeState mirrors upstream serialize_state: strings as-is, anything else json.dumps(ensure_ascii=False).
func SerializeState(state Value) string {
	if state.Kind == KindString {
		return state.Str
	}
	return PyDumps(state, DumpOptions{EnsureASCII: false})
}
