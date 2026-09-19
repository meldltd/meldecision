package laya

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Golden fixtures are produced by export/make_goldens.py from the upstream Python package.
// The tests need an exported checkpoint under $LAYA_MODELS_DIR (default ../models) and
// ONNXRUNTIME_SHARED_LIBRARY_PATH pointing at libonnxruntime; they are skipped otherwise.

type golden struct {
	Model string `json:"model"`
	Cases map[string]struct {
		State     Value `json:"state"`
		Questions Value `json:"questions"`
		Sequences map[string]struct {
			IDs     []int64  `json:"ids"`
			Markers []int    `json:"markers"`
			Options []string `json:"options"`
		} `json:"sequences"`
		Result struct {
			Answers map[string]map[string]json.RawMessage `json:"answers"`
			Usage   Usage                                 `json:"usage"`
		} `json:"result"`
	} `json:"cases"`
}

func loadGolden(t *testing.T, name string) *golden {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "testdata", "golden_"+name+".json"))
	if err != nil {
		t.Skipf("no golden file: %v", err)
	}
	var g golden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	return &g
}

func modelsDir() string {
	if d := os.Getenv("LAYA_MODELS_DIR"); d != "" {
		return d
	}
	return filepath.Join("..", "models")
}

func loadAgent(t *testing.T, name string) *Agent {
	t.Helper()
	dir := filepath.Join(modelsDir(), name)
	if _, err := os.Stat(filepath.Join(dir, "laya_config.json")); err != nil {
		t.Skipf("checkpoint %s not exported (%v)", name, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "model.onnx")); err != nil {
		t.Skipf("checkpoint %s has no model.onnx (%v)", name, err)
	}
	a, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

var goldenModels = []string{"english", "multilingual"}

func TestOptionsAndSequencesMatchPython(t *testing.T) {
	for _, m := range goldenModels {
		t.Run(m, func(t *testing.T) { testSequences(t, m) })
	}
}

func testSequences(t *testing.T, model string) {
	g := loadGolden(t, model)
	dir := filepath.Join(modelsDir(), g.Model)
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Skipf("checkpoint not exported: %v", err)
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer tok.Close()

	for name, c := range g.Cases {
		spec, err := ParseSpec(c.Questions)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		stateText := SerializeState(c.State)
		for i := range spec {
			q := &spec[i]
			want := c.Sequences[q.ID]
			if got := q.Options(); !equalStrings(got, want.Options) {
				t.Errorf("%s/%s options:\n got %q\nwant %q", name, q.ID, got, want.Options)
			}
			seq, err := BuildSequence(tok, cfg, stateText, q)
			if err != nil {
				t.Fatalf("%s/%s: %v", name, q.ID, err)
			}
			if !equalInt64(seq.IDs, want.IDs) {
				t.Errorf("%s/%s token ids differ (len %d vs %d):\n got %v\nwant %v", name, q.ID, len(seq.IDs), len(want.IDs), seq.IDs, want.IDs)
			}
			if !equalInts(seq.Markers, want.Markers) {
				t.Errorf("%s/%s markers: got %v want %v", name, q.ID, seq.Markers, want.Markers)
			}
		}
	}
}

func TestPredictMatchesPython(t *testing.T) {
	for _, m := range goldenModels {
		t.Run(m, func(t *testing.T) { testPredict(t, m) })
	}
}

func testPredict(t *testing.T, model string) {
	g := loadGolden(t, model)
	a := loadAgent(t, g.Model)
	const tol = 2e-3

	for name, c := range g.Cases {
		spec, err := ParseSpec(c.Questions)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		res, err := a.Predict(c.State, spec)
		if err != nil {
			t.Fatalf("%s: predict: %v", name, err)
		}
		if res.Usage.InputTokens != c.Result.Usage.InputTokens {
			t.Errorf("%s: input_tokens got %d want %d", name, res.Usage.InputTokens, c.Result.Usage.InputTokens)
		}
		enc, _ := json.Marshal(res.Answers)
		var got map[string]map[string]json.RawMessage
		if err := json.Unmarshal(enc, &got); err != nil {
			t.Fatal(err)
		}
		for qid, want := range c.Result.Answers {
			g := got[qid]
			if g == nil {
				t.Errorf("%s/%s missing answer", name, qid)
				continue
			}
			for field, wv := range want {
				gv, ok := g[field]
				if !ok {
					t.Errorf("%s/%s missing field %s", name, qid, field)
					continue
				}
				if !approxJSON(gv, wv, tol) {
					t.Errorf("%s/%s.%s:\n got %s\nwant %s", name, qid, field, gv, wv)
				}
			}
		}
		t.Logf("%s: %d questions, %d tokens, %.0f ms inference", name, len(spec), res.Usage.InputTokens, res.Timing.InferenceMs)
	}
}

func TestRoutingAndLanguage(t *testing.T) {
	r, err := NewRouter(RouterOptions{ModelsDir: modelsDir()})
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		state string
		model string
	}{
		{`{"body": "I was charged twice, please refund."}`, "english"},
		{`{"body": "मुझसे दो बार शुल्क लिया गया"}`, "multilingual"},
		{`{"body": "二重に請求されました"}`, "multilingual"},
		{`{"body": "Der Kunde wurde zweimal belastet und möchte das Geld zurück"}`, "multilingual"},
		{`{"body": "12345 !!!"}`, "english"},
		{`"The quick brown fox jumps over the lazy dog"`, "english"},
	}
	for _, c := range cases {
		v, err := ParseValue([]byte(c.state))
		if err != nil {
			t.Fatal(err)
		}
		route, err := r.Route(v, nil, RouteRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if route.Model != c.model {
			t.Errorf("%s: routed to %s (%s), want %s", c.state, route.Model, route.Reason, c.model)
		}
	}
	if rt, _ := r.Route(Value{}, nil, RouteRequest{Model: "ml"}); rt.Model != "multilingual" {
		t.Errorf("alias ml -> %s", rt.Model)
	}
	if rt, _ := r.Route(Value{}, nil, RouteRequest{Lang: "de-AT"}); rt.Model != "multilingual" {
		t.Errorf("lang de-AT -> %s", rt.Model)
	}
	if rt, _ := r.Route(Value{}, nil, RouteRequest{Task: "typed_decisions"}); rt.Model != "typed-decisions" {
		t.Errorf("task typed_decisions -> %s", rt.Model)
	}
	if _, err := r.Route(Value{}, nil, RouteRequest{Model: "nope"}); err == nil {
		t.Error("unknown model should fail")
	}
}

func TestPyDumps(t *testing.T) {
	cases := map[string]string{
		`{"a": 1, "b": [1.0, 2.5e-7, 1e21, true, null], "c": "é\n\"q\"", "d": {"z": 1, "a": 2}}`: `{"a": 1, "b": [1.0, 2.5e-07, 1e+21, true, null], "c": "é\n\"q\"", "d": {"z": 1, "a": 2}}`,
		`[100000000000000000000.0, 0.0001, 0.00001, 123456789012345680000]`:                      `[1e+20, 0.0001, 1e-05, 123456789012345680000]`,
		`"😀"`: `"😀"`,
	}
	for in, want := range cases {
		v, err := ParseValue([]byte(in))
		if err != nil {
			t.Fatal(err)
		}
		if got := PyDumps(v, DumpOptions{}); got != want {
			t.Errorf("PyDumps(%s):\n got %s\nwant %s", in, got, want)
		}
	}
	v, _ := ParseValue([]byte(`{"q": "Grüße 😀"}`))
	if got := PyDumps(v, DumpOptions{EnsureASCII: true}); got != `{"q": "Gr\u00fc\u00dfe \ud83d\ude00"}` {
		t.Errorf("ensure_ascii: %s", got)
	}
}

func TestPresetsParse(t *testing.T) {
	for _, n := range PresetNames() {
		if _, err := Preset(n, nil); err != nil {
			t.Errorf("%s: %v", n, err)
		}
	}
	cats, _ := ParseValue([]byte(`{"a": "x", "b": "y"}`))
	spec, err := Preset("email", &cats)
	if err != nil {
		t.Fatal(err)
	}
	if got := spec[0].Options(); !equalStrings(got, []string{"a: x", "b: y"}) {
		t.Errorf("custom categories: %v", got)
	}
}

func TestCleanEmailBody(t *testing.T) {
	body := "Hi team,\n\nWe were billed twice.\nPlease refund.\n\nThanks,\nBob\n\nOn Mon, Bob wrote:\n> old stuff\n\nThis email is confidential."
	got := CleanEmailBody(body)
	want := "Hi team,\n\nWe were billed twice.\nPlease refund."
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func approxJSON(a, b json.RawMessage, tol float64) bool {
	var fa, fb float64
	if json.Unmarshal(a, &fa) == nil && json.Unmarshal(b, &fb) == nil {
		return math.Abs(fa-fb) <= tol
	}
	var ma, mb map[string]json.RawMessage
	if json.Unmarshal(a, &ma) == nil && json.Unmarshal(b, &mb) == nil {
		if len(ma) != len(mb) {
			return false
		}
		for k, va := range ma {
			vb, ok := mb[k]
			if !ok || !approxJSON(va, vb, tol) {
				return false
			}
		}
		return true
	}
	var sa, sb any
	json.Unmarshal(a, &sa)
	json.Unmarshal(b, &sb)
	ea, _ := json.Marshal(sa)
	eb, _ := json.Marshal(sb)
	return string(ea) == string(eb)
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalInt64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// With MaxLoaded=1 and requests alternating checkpoints, every prediction must still succeed:
// an agent that is being evicted stays alive until in-flight predictions on it finish.
func TestRouterEvictionUnderLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	for _, m := range []string{"english", "multilingual"} {
		if _, err := os.Stat(filepath.Join(modelsDir(), m, "model.onnx")); err != nil {
			t.Skipf("checkpoint %s not exported", m)
		}
	}
	r, err := NewRouter(RouterOptions{ModelsDir: modelsDir(), MaxLoaded: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	spec, _ := Preset("triage", nil)
	states := map[string]Value{
		"english":      StringValue("I was charged twice, please refund me today."),
		"multilingual": StringValue("Mein Konto wurde zweimal belastet, bitte erstatten Sie das Geld."),
	}
	errs := make(chan error, 8)
	for i := 0; i < 4; i++ {
		go func(i int) {
			for j := 0; j < 2; j++ {
				model := []string{"english", "multilingual"}[(i+j)%2]
				res, err := r.Predict(states[model], spec, RouteRequest{Model: model})
				if err == nil && res.Routing.Model != model {
					err = fmt.Errorf("routed to %s, want %s", res.Routing.Model, model)
				}
				if err != nil {
					errs <- err
					return
				}
			}
			errs <- nil
		}(i)
	}
	for i := 0; i < 4; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if n := len(r.Loaded()); n != 1 {
		t.Errorf("loaded=%v, want exactly one resident checkpoint", r.Loaded())
	}
}
