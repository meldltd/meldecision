package laya

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// Port of upstream laya/router.py: pick the checkpoint best suited to a request.
//
//	english          ModernBERT-large, 512 tokens          English text
//	multilingual     mmBERT-base, 1024 tokens              100+ languages
//	typed-decisions  ModernBERT-large, 1024 tokens         the four typed-decisions workflows
//
// Script detection is the primary signal: the English checkpoint collapses to near-random on
// non-Latin scripts while still reporting high confidence.

// ModelNames are the canonical checkpoint names, which are also the directory names under the models root.
var ModelNames = []string{"english", "multilingual", "typed-decisions"}

var aliases = map[string]string{
	"en": "english", "laya": "english", "default": "english",
	"multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
	"typed": "typed-decisions", "typed_decisions": "typed-decisions",
	"laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

// NormaliseName resolves aliases to a canonical checkpoint name.
func NormaliseName(name string) (string, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if a, ok := aliases[key]; ok {
		key = a
	}
	for _, n := range ModelNames {
		if n == key {
			return key, nil
		}
	}
	al := make([]string, 0, len(aliases))
	for k := range aliases {
		al = append(al, k)
	}
	sort.Strings(al)
	return "", fmt.Errorf("unknown model %q; choose one of %v (or an alias: %v)", name, ModelNames, al)
}

// Question-id signatures of the four typed-decisions workflows (exact set match required).
var typedDecisionWorkflows = map[string][]string{
	"agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
	"customer_service":          {"action", "category", "churn_risk", "needs_human", "urgency"},
	"invoice_processing":        {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
	"security_incidents":        {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}

// MatchTypedDecisionsWorkflow names the workflow whose question ids these are, else "".
func MatchTypedDecisionsWorkflow(spec Spec) string {
	ids := map[string]bool{}
	for _, q := range spec {
		ids[q.ID] = true
	}
	names := make([]string, 0, len(typedDecisionWorkflows))
	for n := range typedDecisionWorkflows {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, wf := range names {
		sig := typedDecisionWorkflows[wf]
		if len(sig) != len(ids) {
			continue
		}
		ok := true
		for _, s := range sig {
			if !ids[s] {
				ok = false
				break
			}
		}
		if ok {
			return wf
		}
	}
	return ""
}

// Route is the routing outcome: which model, why, and what was detected.
type Route struct {
	Model     string     `json:"model"`
	Repo      string     `json:"repo"`
	Reason    string     `json:"reason"`
	Detection *Detection `json:"detection"`
	Workflow  *string    `json:"workflow"`
}

// RouteRequest are the optional routing hints a caller may pass with a request.
type RouteRequest struct {
	Model string // explicit checkpoint (name or alias)
	Task  string // "typed_decisions" (or a model name) selects a checkpoint by task
	Lang  string // BCP-47-ish language code; "en" routes to english, anything else to multilingual
}

// RouterOptions configure a Router.
type RouterOptions struct {
	// ModelsDir holds one sub-directory per checkpoint (models/english, models/multilingual, ...).
	ModelsDir string
	// Default is used when nothing can be detected (default "english").
	Default string
	// MaxLoaded caps resident checkpoints; least-recently-used is evicted (default: all).
	MaxLoaded int
	// AutoTaskDetection routes exact typed-decisions question-id sets to that checkpoint.
	AutoTaskDetection bool
	// Load options applied to every checkpoint.
	Load Options
}

// Router lazily loads checkpoints and sends each request to the right one.
type Router struct {
	opt    RouterOptions
	mu     sync.Mutex // guards agents/order
	loadMu sync.Mutex // serialises checkpoint loads so a burst of requests loads a model once
	agents map[string]*Agent
	order  []string // least-recently-used first
}

// NewRouter creates a router over opt.ModelsDir. Nothing is loaded until needed (or Preload).
func NewRouter(opt RouterOptions) (*Router, error) {
	if opt.ModelsDir == "" {
		opt.ModelsDir = "models"
	}
	if opt.Default == "" {
		opt.Default = "english"
	}
	d, err := NormaliseName(opt.Default)
	if err != nil {
		return nil, err
	}
	opt.Default = d
	if opt.MaxLoaded <= 0 {
		opt.MaxLoaded = len(ModelNames)
	}
	return &Router{opt: opt, agents: map[string]*Agent{}}, nil
}

// Dir is the checkpoint directory for a model name.
func (r *Router) Dir(name string) string { return filepath.Join(r.opt.ModelsDir, name) }

// Available lists the checkpoints that exist on disk (exported), in canonical order.
func (r *Router) Available() []string {
	var out []string
	for _, n := range ModelNames {
		if _, err := os.Stat(filepath.Join(r.Dir(n), "laya_config.json")); err == nil {
			out = append(out, n)
		}
	}
	return out
}

// Loaded lists resident checkpoints, least-recently-used first.
func (r *Router) Loaded() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.order...)
}

// Get returns the Agent for name, loading it on first use.
//
// The returned agent may be evicted (and closed) by a later Get once MaxLoaded is exceeded;
// Agent.Predict fails cleanly with "agent is closed" in that case. Router.Predict uses acquire
// instead, which pins the agent for the duration of the call.
func (r *Router) Get(name string) (*Agent, error) {
	a, release, err := r.acquire(name)
	if err != nil {
		return nil, err
	}
	release()
	return a, nil
}

// acquire returns the agent with its read lock held; release must be called when done.
func (r *Router) acquire(name string) (*Agent, func(), error) {
	key, err := NormaliseName(name)
	if err != nil {
		return nil, nil, err
	}
	r.mu.Lock()
	if a, ok := r.agents[key]; ok {
		r.touch(key)
		a.mu.RLock()
		r.mu.Unlock()
		return a, a.mu.RUnlock, nil
	}
	r.mu.Unlock()

	r.loadMu.Lock()
	defer r.loadMu.Unlock()
	r.mu.Lock()
	if a, ok := r.agents[key]; ok { // loaded while we waited
		r.touch(key)
		a.mu.RLock()
		r.mu.Unlock()
		return a, a.mu.RUnlock, nil
	}
	r.mu.Unlock()

	a, err := Load(r.Dir(key), r.opt.Load)
	if err != nil {
		return nil, nil, fmt.Errorf("load %s: %w", key, err)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.agents[key] = a
	r.order = append(r.order, key)
	a.mu.RLock() // the new agent is most-recently-used, so evict() never picks it
	r.evict()
	return a, a.mu.RUnlock, nil
}

// SetMaxLoaded changes the LRU cap (raising it never evicts; lowering it evicts on the next load).
func (r *Router) SetMaxLoaded(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.opt.MaxLoaded = max(1, n)
}

func (r *Router) touch(key string) {
	for i, k := range r.order {
		if k == key {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
	r.order = append(r.order, key)
}

func (r *Router) evict() {
	for len(r.order) > r.opt.MaxLoaded {
		victim := r.order[0]
		r.order = r.order[1:]
		if a, ok := r.agents[victim]; ok {
			delete(r.agents, victim)
			// Close later: in-flight Run() calls on this agent hold no lock, so give them a
			// moment. ORT's session is reference-safe for Destroy after Run returns.
			go a.Close()
		}
	}
}

// Preload loads the named checkpoints (all available ones when names is empty) up front.
func (r *Router) Preload(names []string) error {
	if len(names) == 0 {
		names = r.Available()
	}
	r.mu.Lock()
	if r.opt.MaxLoaded < len(names) {
		r.opt.MaxLoaded = len(names)
	}
	r.mu.Unlock()
	for _, n := range names {
		if _, err := r.Get(n); err != nil {
			return err
		}
	}
	return nil
}

// Close frees every loaded checkpoint.
func (r *Router) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for k, a := range r.agents {
		a.Close()
		delete(r.agents, k)
	}
	r.order = nil
}

func (r *Router) repo(name string) string {
	if cfg, err := LoadConfig(r.Dir(name)); err == nil && cfg.Repo != "" {
		return cfg.Repo
	}
	sub := map[string]string{"english": "", "multilingual": "/multilingual", "typed-decisions": "/typed-decisions"}
	return "convaiinnovations/laya" + sub[name]
}

// Route decides which checkpoint to use without loading or running anything.
// Precedence: explicit Model > explicit Task > detected workflow (opt-in) > explicit Lang >
// detected script/language > default.
func (r *Router) Route(state Value, spec Spec, req RouteRequest) (Route, error) {
	if req.Model != "" {
		key, err := NormaliseName(req.Model)
		if err != nil {
			return Route{}, err
		}
		return Route{Model: key, Repo: r.repo(key), Reason: "explicit model=" + pyRepr(req.Model)}, nil
	}
	if req.Task != "" {
		t := req.Task
		if strings.ReplaceAll(strings.ToLower(t), "-", "_") == "typed_decisions" {
			t = "typed-decisions"
		}
		key, err := NormaliseName(t)
		if err != nil {
			return Route{}, err
		}
		return Route{Model: key, Repo: r.repo(key), Reason: "explicit task=" + pyRepr(req.Task)}, nil
	}
	var workflow *string
	if wf := MatchTypedDecisionsWorkflow(spec); wf != "" {
		workflow = &wf
		if r.opt.AutoTaskDetection {
			return Route{Model: "typed-decisions", Repo: r.repo("typed-decisions"),
				Reason: "question ids match the " + pyRepr(wf) + " typed-decisions workflow", Workflow: workflow}, nil
		}
	}
	if req.Lang != "" {
		base := strings.ToLower(strings.SplitN(req.Lang, "-", 2)[0])
		key := "multilingual"
		if base == "en" || base == "eng" || base == "english" {
			key = "english"
		}
		return Route{Model: key, Repo: r.repo(key), Reason: "explicit lang=" + pyRepr(req.Lang), Workflow: workflow}, nil
	}
	det := Analyse(state)
	var key, reason string
	switch {
	case det.Script == "unknown":
		key = r.opt.Default
		reason = fmt.Sprintf("no letters detected in state; using default (%s)", key)
	case det.Script != "latin":
		key = "multilingual"
		reason = fmt.Sprintf("non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it",
			det.Script, 100*det.NonLatinFraction)
	case !det.IsEnglish:
		key = "multilingual"
		reason = "Latin script but language looks like " + pyRepr(*det.Language) + ", not English"
	default:
		key = "english"
		reason = "English Latin text"
	}
	return Route{Model: key, Repo: r.repo(key), Reason: reason, Detection: &det, Workflow: workflow}, nil
}

// Predict routes, then answers every question in one forward pass on the chosen checkpoint.
func (r *Router) Predict(state Value, spec Spec, req RouteRequest) (*Result, error) {
	route, err := r.Route(state, spec, req)
	if err != nil {
		return nil, err
	}
	agent, release, err := r.acquire(route.Model)
	if err != nil {
		return nil, err
	}
	defer release()
	res, err := agent.predictLocked(state, spec)
	if err != nil {
		return nil, err
	}
	res.Routing = &route
	return res, nil
}

// pyRepr quotes a string the way Python's repr() does for the routing reason text.
func pyRepr(s string) string {
	if strings.Contains(s, "'") && !strings.Contains(s, "\"") {
		return "\"" + s + "\""
	}
	return "'" + strings.ReplaceAll(s, "'", "\\'") + "'"
}
