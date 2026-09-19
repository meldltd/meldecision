package main

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"meld.si/laya/laya"
)

// These tests exercise request validation, routing and presets without any model loaded.
func TestValidationAndRouting(t *testing.T) {
	r, err := laya.NewRouter(laya.RouterOptions{ModelsDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	var ready atomic.Bool
	app := newApp(r, &ready, 1<<20, 0)

	do := func(method, path, body string) (int, map[string]any) {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := app.Test(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		var out map[string]any
		_ = json.Unmarshal(raw, &out)
		return resp.StatusCode, out
	}

	if code, _ := do("GET", "/healthz", ""); code != 200 {
		t.Errorf("healthz %d", code)
	}
	if code, _ := do("GET", "/readyz", ""); code != 503 {
		t.Errorf("readyz before preload %d", code)
	}
	ready.Store(true)
	if code, _ := do("GET", "/readyz", ""); code != 200 {
		t.Errorf("readyz after preload %d", code)
	}
	if code, out := do("POST", "/v1/predict", `{"spec": {}}`); code != 400 || !strings.Contains(out["error"].(string), "context") {
		t.Errorf("missing context: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x"}`); code != 400 || !strings.Contains(out["error"].(string), "spec") {
		t.Errorf("missing spec: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x", "spec": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}}`); code != 400 || !strings.Contains(out["error"].(string), "two rubric") {
		t.Errorf("bad score: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x", "bogus": 1, "spec": {}}`); code != 400 || !strings.Contains(out["error"].(string), "unknown field") {
		t.Errorf("unknown field: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x", "spec": {"q": {"type": "noul", "instructions": "?"}}, "model": "nope"}`); code != 400 || !strings.Contains(out["error"].(string), "unknown model") {
		t.Errorf("unknown model: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x", "spec": {}, "max_len": "big"}`); code != 400 || !strings.Contains(out["error"].(string), "max_len") {
		t.Errorf("non-numeric max_len: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x", "spec": {}, "max_len": 16384}`); code != 400 || !strings.Contains(out["error"].(string), "max_len") {
		t.Errorf("max_len above the encoder limit: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/predict", `{"context": "x", "spec": {}, "max_len": 2.5}`); code != 400 || !strings.Contains(out["error"].(string), "max_len") {
		t.Errorf("fractional max_len: %d %v", code, out)
	}
	// a valid max_len passes validation; with no checkpoint on disk it then fails to load -> 503
	if code, _ := do("POST", "/v1/predict", `{"context": "x", "spec": {"q": {"type": "noul", "instructions": "?"}}, "max_len": 4096}`); code != 503 {
		t.Errorf("valid max_len should reach model loading (503), got %d", code)
	}
	// no checkpoint on disk -> 503, not 500
	if code, _ := do("POST", "/v1/predict", `{"context": "x", "spec": {"q": {"type": "noul", "instructions": "?"}}}`); code != 503 {
		t.Errorf("missing checkpoint should be 503, got %d", code)
	}
	if code, out := do("POST", "/v1/route", `{"context": {"body": "二重に請求されました"}}`); code != 200 || out["model"] != "multilingual" {
		t.Errorf("route: %d %v", code, out)
	}
	if code, out := do("POST", "/v1/route", `{"context": "hello there my friend", "lang": "fr"}`); code != 200 || out["model"] != "multilingual" {
		t.Errorf("route lang: %d %v", code, out)
	}
	if code, out := do("GET", "/v1/presets", ""); code != 200 || len(out["presets"].([]any)) != 5 {
		t.Errorf("presets: %d %v", code, out)
	}
	if code, out := do("GET", "/v1/presets/guard", ""); code != 200 || out["name"] != "guard" {
		t.Errorf("preset guard: %d %v", code, out)
	}
	if code, _ := do("GET", "/v1/presets/nope", ""); code != 404 {
		t.Errorf("preset nope: %d", code)
	}
	if code, out := do("POST", "/v1/presets/triage", `{"context": {"message": "hi"}, "spec": {}}`); code != 400 || !strings.Contains(out["error"].(string), "own spec") {
		t.Errorf("preset with spec: %d %v", code, out)
	}
	if code, out := do("GET", "/v1/models", ""); code != 200 || out["ready"] != true {
		t.Errorf("models: %d %v", code, out)
	}
}
