package laya

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// Options tune how a checkpoint is loaded.
type Options struct {
	// IntraOpThreads is ONNX Runtime's per-op parallelism (0 = number of CPUs).
	IntraOpThreads int
	// CoreML enables the CoreML execution provider on macOS (unsupported ops fall back to CPU).
	CoreML bool
	// ModelFile overrides the graph file name inside the checkpoint directory
	// (e.g. "model.int8.onnx" for the dynamically quantised variant).
	ModelFile string
	// MaxLen overrides the checkpoint's max_len (context window in tokens) for every
	// request, e.g. 2048 or 4096. 0 keeps the value from laya_config.json (512/1024).
	// Must be at most MaxContext. Longer contexts cost more memory and time per request
	// and were not part of fine-tuning, so probabilities may be slightly less calibrated.
	MaxLen int
}

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// InitRuntime initialises ONNX Runtime once. libPath is the onnxruntime shared library
// (libonnxruntime.so / .dylib); empty uses $ONNXRUNTIME_SHARED_LIBRARY_PATH or the loader default.
func InitRuntime(libPath string) error {
	ortInitOnce.Do(func() {
		if libPath == "" {
			libPath = os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
		}
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		if !ort.IsInitialized() {
			ortInitErr = ort.InitializeEnvironment(ort.WithLogLevelError())
		}
	})
	return ortInitErr
}

// Agent is one loaded Laya checkpoint: tokenizer + ONNX session + calibration config.
type Agent struct {
	Dir     string
	Config  *Config
	tok     *Tokenizer
	session *ort.DynamicAdvancedSession
	mu      sync.RWMutex // ORT Run is thread-safe; readers run concurrently, Close takes the write lock
	closed  bool
}

// Load opens the checkpoint directory produced by export/export_onnx.py.
func Load(dir string, opt Options) (*Agent, error) {
	if err := InitRuntime(""); err != nil {
		return nil, fmt.Errorf("onnxruntime: %w", err)
	}
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	if err := cfg.CheckMaxLen(opt.MaxLen); err != nil {
		return nil, fmt.Errorf("%s: %w", dir, err)
	}
	cfg = cfg.withMaxLen(opt.MaxLen)
	tok, err := LoadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	file := cfg.ONNX.File
	if opt.ModelFile != "" {
		file = opt.ModelFile
	}
	modelPath := filepath.Join(dir, file)
	if _, err := os.Stat(modelPath); err != nil {
		tok.Close()
		return nil, fmt.Errorf("model file %s: %w", modelPath, err)
	}

	so, err := ort.NewSessionOptions()
	if err != nil {
		tok.Close()
		return nil, err
	}
	defer so.Destroy()
	threads := opt.IntraOpThreads
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	_ = so.SetIntraOpNumThreads(threads)
	_ = so.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
	if opt.CoreML && runtime.GOOS == "darwin" {
		if err := so.AppendExecutionProviderCoreMLV2(map[string]string{"ModelFormat": "MLProgram"}); err != nil {
			return nil, fmt.Errorf("enable CoreML: %w", err)
		}
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath, cfg.ONNX.Inputs, cfg.ONNX.Outputs, so)
	if err != nil {
		tok.Close()
		return nil, fmt.Errorf("open ONNX session %s: %w", modelPath, err)
	}
	return &Agent{Dir: dir, Config: cfg, tok: tok, session: sess}, nil
}

// Close frees the native session and tokenizer.
func (a *Agent) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
	err := a.session.Destroy()
	if e := a.tok.Close(); err == nil {
		err = e
	}
	return err
}

// Tokenizer exposes the checkpoint's tokenizer (used by tests and tooling).
func (a *Agent) Tokenizer() *Tokenizer { return a.tok }

// Result is the upstream `system_one` payload.
type Result struct {
	Model   string  `json:"model"`
	Answers Answers `json:"answers"`
	Usage   Usage   `json:"usage"`
	Routing *Route  `json:"routing,omitempty"`
	Timing  *Timing `json:"timing,omitempty"`
}

// Usage mirrors the upstream usage block.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Timing reports where the milliseconds went.
type Timing struct {
	TokenizeMs  float64 `json:"tokenize_ms"`
	InferenceMs float64 `json:"inference_ms"`
	TotalMs     float64 `json:"total_ms"`
}

// Predict evaluates every question in spec over state in one forward pass.
// It holds the agent's read lock for the whole call, so Close waits for it.
func (a *Agent) Predict(state Value, spec Spec) (*Result, error) {
	return a.PredictMaxLen(state, spec, 0)
}

// PredictMaxLen is Predict with a per-call context window: maxLen tokens (e.g. 2048 or
// 4096) instead of the agent's configured max_len; 0 uses the configured value.
func (a *Agent) PredictMaxLen(state Value, spec Spec, maxLen int) (*Result, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.predictLocked(state, spec, maxLen)
}

// predictLocked is PredictMaxLen for callers that already hold a.mu.RLock (the Router).
func (a *Agent) predictLocked(state Value, spec Spec, maxLen int) (*Result, error) {
	if a.closed {
		return nil, fmt.Errorf("agent is closed")
	}
	if err := a.Config.CheckMaxLen(maxLen); err != nil {
		return nil, err
	}
	cfg := a.Config.withMaxLen(maxLen)
	t0 := time.Now()
	stateText := SerializeState(state)
	seqs := make([]Sequence, len(spec))
	for i := range spec {
		s, err := BuildSequence(a.tok, cfg, stateText, &spec[i])
		if err != nil {
			return nil, err
		}
		seqs[i] = s
	}
	b := Collate(seqs, a.Config.SpecialTokens.PadID)
	t1 := time.Now()

	logits, act, nAct, err := a.run(b)
	if err != nil {
		return nil, err
	}
	t2 := time.Now()

	answers := make(Answers, len(spec))
	for r := range spec {
		q := &spec[r]
		k := len(seqs[r].Markers)
		row := logits[r*b.K : r*b.K+k]
		p := scaledSoftmax(row, a.Config.temperatureFor(q.Type, k))
		actRow := act[r*nAct : (r+1)*nAct]
		actProb := round4(float64(softmax(actRow)[0]))
		answers[r] = formatAnswer(q, p, actProb)
	}
	t3 := time.Now()
	return &Result{
		Model:   a.Config.Name,
		Answers: answers,
		Usage:   Usage{InputTokens: b.NumTokens},
		Timing: &Timing{
			TokenizeMs:  ms(t1.Sub(t0)),
			InferenceMs: ms(t2.Sub(t1)),
			TotalMs:     ms(t3.Sub(t0)),
		},
	}, nil
}

func ms(d time.Duration) float64 { return math.Round(float64(d)/float64(time.Millisecond)*100) / 100 }

// run executes the graph and returns logits [N,K] and act logits [N,nAct] as flat slices.
func (a *Agent) run(b *Batch) (logits []float32, act []float32, nAct int, err error) {
	mk := func(shape ort.Shape, data []int64) (*ort.Tensor[int64], error) {
		return ort.NewTensor(shape, data)
	}
	inIDs, err := mk(ort.NewShape(int64(b.N), int64(b.L)), b.InputIDs)
	if err != nil {
		return nil, nil, 0, err
	}
	defer inIDs.Destroy()
	att, err := mk(ort.NewShape(int64(b.N), int64(b.L)), b.AttentionMask)
	if err != nil {
		return nil, nil, 0, err
	}
	defer att.Destroy()
	mpos, err := mk(ort.NewShape(int64(b.N), int64(b.K)), b.MarkerPos)
	if err != nil {
		return nil, nil, 0, err
	}
	defer mpos.Destroy()
	mmask, err := mk(ort.NewShape(int64(b.N), int64(b.K)), b.MarkerMask)
	if err != nil {
		return nil, nil, 0, err
	}
	defer mmask.Destroy()
	qt, err := mk(ort.NewShape(int64(b.N)), b.QType)
	if err != nil {
		return nil, nil, 0, err
	}
	defer qt.Destroy()

	outputs := []ort.Value{nil, nil}
	if err := a.session.Run([]ort.Value{inIDs, att, mpos, mmask, qt}, outputs); err != nil {
		return nil, nil, 0, fmt.Errorf("onnxruntime run: %w", err)
	}
	for _, o := range outputs {
		if o != nil {
			defer o.Destroy()
		}
	}
	lt, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, 0, fmt.Errorf("unexpected logits output type %T", outputs[0])
	}
	at, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, 0, fmt.Errorf("unexpected act_logits output type %T", outputs[1])
	}
	ls := lt.GetShape()
	as := at.GetShape()
	if len(ls) != 2 || int(ls[0]) != b.N || int(ls[1]) != b.K {
		return nil, nil, 0, fmt.Errorf("unexpected logits shape %v (want [%d %d])", ls, b.N, b.K)
	}
	if len(as) != 2 || int(as[0]) != b.N {
		return nil, nil, 0, fmt.Errorf("unexpected act_logits shape %v", as)
	}
	logits = append([]float32(nil), lt.GetData()...)
	act = append([]float32(nil), at.GetData()...)
	return logits, act, int(as[1]), nil
}

// ---------------------------------------------------------------- post-processing (upstream system_one)
//
// Upstream does this in NumPy on float32 logits; the arithmetic below mirrors that precision so
// that 4-decimal outputs match.

func scaledSoftmax(z []float32, temperature float64) []float32 {
	t := float32(math.Max(1e-3, temperature))
	out := make([]float32, len(z))
	mx := float32(math.Inf(-1))
	for i, v := range z {
		out[i] = v / t
		if out[i] > mx {
			mx = out[i]
		}
	}
	var sum float32
	for i := range out {
		out[i] = float32(math.Exp(float64(out[i] - mx)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func softmax(z []float32) []float32 { return scaledSoftmax(z, 1) }

// confidenceFromProbs is 1 - H(p)/log(k), clipped to [0, 1].
func confidenceFromProbs(p []float32, k int) float64 {
	if k < 2 {
		return 1
	}
	var ent float32
	for _, v := range p[:k] {
		c := v
		if c < 1e-12 {
			c = 1e-12
		}
		if c > 1 {
			c = 1
		}
		ent -= v * float32(math.Log(float64(c)))
	}
	conf := float32(1) - ent/float32(math.Log(float64(k)))
	return math.Min(math.Max(float64(conf), 0), 1)
}

// round4 rounds to 4 decimals like Python's round(x, 4) (correctly rounded from the binary value).
func round4(x float64) float64 {
	v, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)
	if err != nil {
		return x
	}
	return v
}

// Answer is one question's result. Exactly one of Choice / Score / Noul is set by Type.
type Answer struct {
	ID            string
	Type          QType
	Choice        string
	Score         float64
	Legend        []Value // score: the raw rubric levels
	Noul          float64
	ProbKeys      []string
	Probabilities []float64
	Confidence    float64
	ActProb       float64
}

// Answers is the ordered list of answers; it serialises as an object keyed by question id.
type Answers []Answer

func formatAnswer(q *Question, p []float32, actProb float64) Answer {
	k := len(p)
	a := Answer{ID: q.ID, Type: q.Type, ActProb: actProb, Confidence: round4(confidenceFromProbs(p, k))}
	rounded := make([]float64, k)
	for i, v := range p {
		rounded[i] = round4(float64(v))
	}
	switch q.Type {
	case QChoice:
		best := 0
		for i := range p {
			if p[i] > p[best] {
				best = i
			}
		}
		a.Choice = q.Keys[best]
		a.ProbKeys = q.Keys
		a.Probabilities = rounded
	case QScore:
		exp := 0.0 // np.arange(k) * p promotes to float64 upstream
		for i, v := range p {
			exp += float64(i) * float64(v)
		}
		a.Score = round4(exp)
		a.Legend = q.Levels
		a.ProbKeys = make([]string, k)
		for i := range a.ProbKeys {
			a.ProbKeys[i] = strconv.Itoa(i)
		}
		a.Probabilities = rounded
	default:
		p1 := float64(p[1])
		a.Noul = round4(p1)
		a.Confidence = round4(math.Max(p1, 1-p1))
	}
	return a
}

// MarshalJSON renders the upstream answer shape.
func (a Answer) MarshalJSON() ([]byte, error) {
	obj := Value{Kind: KindObject}
	add := func(k string, v Value) { obj.Obj = append(obj.Obj, Field{Key: k, Val: v}) }
	num := func(f float64) Value { return Value{Kind: KindNumber, Num: pyFloatRepr(f)} } // 1.0 stays a float, as in Python
	probs := func() Value {
		o := Value{Kind: KindObject}
		for i, k := range a.ProbKeys {
			o.Obj = append(o.Obj, Field{Key: k, Val: num(a.Probabilities[i])})
		}
		return o
	}
	add("type", StringValue(a.Type.String()))
	switch a.Type {
	case QChoice:
		add("choice", StringValue(a.Choice))
		add("probabilities", probs())
	case QScore:
		add("score", num(a.Score))
		legend := Value{Kind: KindObject}
		for i, l := range a.Legend {
			legend.Obj = append(legend.Obj, Field{Key: strconv.Itoa(i), Val: l})
		}
		add("legend", legend)
		add("probabilities", probs())
	default:
		add("noul", num(a.Noul))
	}
	add("confidence", num(a.Confidence))
	add("action", Value{Kind: KindObject, Obj: []Field{{Key: "act_probability", Val: num(a.ActProb)}}})
	return obj.MarshalJSON()
}

// MarshalJSON renders answers as {"<id>": {...}, ...} in request order.
func (as Answers) MarshalJSON() ([]byte, error) {
	obj := Value{Kind: KindObject}
	for _, a := range as {
		enc, err := a.MarshalJSON()
		if err != nil {
			return nil, err
		}
		v, err := ParseValue(enc)
		if err != nil {
			return nil, err
		}
		obj.Obj = append(obj.Obj, Field{Key: a.ID, Val: v})
	}
	return obj.MarshalJSON()
}

// RuntimeVersion reports the ONNX Runtime library version once InitRuntime succeeded.
func RuntimeVersion() string { return ort.GetVersion() }
