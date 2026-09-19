// Command layad serves Laya typed decisions over HTTP (GoFiber + ONNX Runtime).
//
//	POST /v1/predict          {"context": <any JSON>, "spec": {<qid>: {...}}, "model"?, "lang"?, "task"?, "max_len"?}
//	POST /v1/presets/:name    {"context": <any JSON>, "model"?, "lang"?, "categories"? (email only), "max_len"?}
//	POST /v1/route            {"context": ..., "spec"?, "model"?, "lang"?, "task"?}  -> routing decision only
//	GET  /v1/presets          list presets;  GET /v1/presets/:name -> the preset's spec
//	GET  /v1/models           available / loaded checkpoints
//	GET  /healthz             liveness;  GET /readyz -> 200 once the preload finished
//
// Configuration (flags override environment variables):
//
//	-addr            LAYA_ADDR              listen address (default :8080)
//	-models          LAYA_MODELS_DIR        directory with one sub-directory per checkpoint (default ./models)
//	-preload         LAYA_PRELOAD           comma list of checkpoints to load at start ("all", "none", or names; default all)
//	-max-loaded      LAYA_MAX_LOADED        LRU cap on resident checkpoints (default: number preloaded, min 1)
//	-default-model   LAYA_DEFAULT_MODEL     checkpoint used when nothing can be detected (default english)
//	-auto-task       LAYA_AUTO_TASK         route exact typed-decisions question sets to that checkpoint (default false)
//	-threads         LAYA_THREADS           ONNX Runtime intra-op threads (default: all CPUs)
//	-model-file      LAYA_MODEL_FILE        graph file inside each checkpoint dir (default model.onnx; e.g. model.int8.onnx)
//	-coreml          LAYA_COREML            macOS: enable the CoreML execution provider (default false)
//	-ort             ONNXRUNTIME_SHARED_LIBRARY_PATH  path to libonnxruntime.{so,dylib}
//	-body-limit      LAYA_BODY_LIMIT        max request body in bytes (default 4 MiB)
//	-max-len         LAYA_MAX_LEN           context window in tokens for every checkpoint, e.g. 2048 or 4096
//	                                        (default 0 = each checkpoint's own max_len: 512 / 1024; max 8192)
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
	fiberrecover "github.com/gofiber/fiber/v3/middleware/recover"

	"meld.si/laya/laya"
)

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		b, err := strconv.ParseBool(v)
		if err == nil {
			return b
		}
	}
	return def
}

func main() {
	addr := flag.String("addr", envStr("LAYA_ADDR", ":8080"), "listen address")
	modelsDir := flag.String("models", envStr("LAYA_MODELS_DIR", "models"), "checkpoint root directory")
	preload := flag.String("preload", envStr("LAYA_PRELOAD", "all"), "checkpoints to load at start: all | none | a,b")
	maxLoaded := flag.Int("max-loaded", envInt("LAYA_MAX_LOADED", 0), "LRU cap on resident checkpoints (0 = number preloaded)")
	defModel := flag.String("default-model", envStr("LAYA_DEFAULT_MODEL", "english"), "checkpoint used when nothing can be detected")
	autoTask := flag.Bool("auto-task", envBool("LAYA_AUTO_TASK", false), "route typed-decisions question sets automatically")
	threads := flag.Int("threads", envInt("LAYA_THREADS", 0), "ONNX Runtime intra-op threads (0 = all CPUs)")
	modelFile := flag.String("model-file", envStr("LAYA_MODEL_FILE", ""), "graph file name inside each checkpoint directory")
	coreml := flag.Bool("coreml", envBool("LAYA_COREML", false), "enable the CoreML execution provider (macOS)")
	ortLib := flag.String("ort", os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"), "path to the onnxruntime shared library")
	bodyLimit := flag.Int("body-limit", envInt("LAYA_BODY_LIMIT", 4<<20), "max request body size in bytes")
	maxLen := flag.Int("max-len", envInt("LAYA_MAX_LEN", 0), "context window in tokens for every checkpoint, e.g. 2048 or 4096 (0 = checkpoint default, max 8192)")
	flag.Parse()

	if *maxLen < 0 || *maxLen > laya.MaxContext {
		log.Fatalf("-max-len %d: must be between 0 (checkpoint default) and %d", *maxLen, laya.MaxContext)
	}

	if err := laya.InitRuntime(*ortLib); err != nil {
		log.Fatalf("onnxruntime: %v (set -ort / ONNXRUNTIME_SHARED_LIBRARY_PATH to libonnxruntime)", err)
	}

	router, err := laya.NewRouter(laya.RouterOptions{
		ModelsDir:         *modelsDir,
		Default:           *defModel,
		MaxLoaded:         *maxLoaded,
		AutoTaskDetection: *autoTask,
		Load:              laya.Options{IntraOpThreads: *threads, CoreML: *coreml, ModelFile: *modelFile, MaxLen: *maxLen},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer router.Close()

	avail := router.Available()
	if len(avail) == 0 {
		log.Fatalf("no exported checkpoints under %s (run: python export/export_onnx.py all)", *modelsDir)
	}
	log.Printf("laya: %d CPUs, onnxruntime %s, checkpoints available: %s", runtime.NumCPU(), ortVersion(), strings.Join(avail, ", "))
	if *maxLen > 0 {
		log.Printf("laya: context window overridden to %d tokens for every checkpoint (-max-len)", *maxLen)
	}

	var ready atomic.Bool
	var names []string
	switch strings.ToLower(strings.TrimSpace(*preload)) {
	case "all", "":
		names = avail
	case "none":
	default:
		names = strings.Split(*preload, ",")
	}
	if n := len(names); *maxLoaded > 0 && *maxLoaded < n {
		log.Printf("-max-loaded %d is below the %d preloaded checkpoints; raising it to %d", *maxLoaded, n, n)
		router.SetMaxLoaded(n)
	}
	go func() {
		for _, n := range names {
			n = strings.TrimSpace(n)
			if n == "" {
				continue
			}
			t := time.Now()
			if _, err := router.Get(n); err != nil {
				log.Printf("preload %s: %v", n, err)
				continue
			}
			log.Printf("loaded %s in %.1fs", n, time.Since(t).Seconds())
		}
		ready.Store(true)
	}()

	app := newApp(router, &ready, *bodyLimit, *maxLen)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("listening on %s", *addr)
	if err := app.Listen(*addr, fiber.ListenConfig{GracefulContext: ctx, DisableStartupMessage: true}); err != nil {
		log.Fatal(err)
	}
}

// newApp wires the HTTP routes onto a Fiber app. maxLen is the -max-len override (0 = none).
func newApp(router *laya.Router, ready *atomic.Bool, bodyLimit int, maxLen int) *fiber.App {
	app := fiber.New(fiber.Config{
		AppName:      "layad",
		BodyLimit:    bodyLimit,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 120 * time.Second,
		ErrorHandler: errorHandler,
	})
	app.Use(fiberrecover.New())
	app.Use(logger.New(logger.Config{Format: "${time} ${status} ${method} ${path} ${latency}\n"}))

	s := &server{router: router, ready: ready, maxLen: maxLen}
	app.Get("/healthz", func(c fiber.Ctx) error { return c.JSON(fiber.Map{"status": "ok"}) })
	app.Get("/readyz", func(c fiber.Ctx) error {
		if !ready.Load() {
			return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"status": "loading", "loaded": router.Loaded()})
		}
		return c.JSON(fiber.Map{"status": "ready", "loaded": router.Loaded()})
	})
	v1 := app.Group("/v1")
	v1.Get("/models", s.models)
	v1.Post("/predict", s.predict)
	v1.Post("/route", s.route)
	v1.Get("/presets", s.listPresets)
	v1.Get("/presets/:name", s.getPreset)
	v1.Post("/presets/:name", s.runPreset)
	return app
}

func ortVersion() string {
	defer func() { _ = recover() }()
	return laya.RuntimeVersion()
}

// ---------------------------------------------------------------- handlers

type server struct {
	router *laya.Router
	ready  *atomic.Bool
	maxLen int // -max-len override (0 = checkpoint default), reported by /v1/models
}

type apiError struct {
	Status  int
	Message string
}

func (e *apiError) Error() string { return e.Message }

func badRequest(format string, args ...any) error {
	return &apiError{Status: fiber.StatusBadRequest, Message: fmt.Sprintf(format, args...)}
}

func errorHandler(c fiber.Ctx, err error) error {
	var ae *apiError
	if errors.As(err, &ae) {
		return c.Status(ae.Status).JSON(fiber.Map{"error": ae.Message})
	}
	var fe *fiber.Error
	if errors.As(err, &fe) {
		return c.Status(fe.Code).JSON(fiber.Map{"error": fe.Message})
	}
	log.Printf("error: %v", err)
	return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
}

// request is the JSON body shared by /v1/predict, /v1/route and /v1/presets/:name.
// "context"/"state" and "spec"/"questions" are accepted interchangeably.
type request struct {
	Context    laya.Value
	Spec       *laya.Value
	Categories *laya.Value
	Route      laya.RouteRequest
}

func parseRequest(c fiber.Ctx) (*request, error) {
	body := c.Body()
	if len(body) == 0 {
		return nil, badRequest("empty request body; send JSON with \"context\" and \"spec\"")
	}
	v, err := laya.ParseValue(body)
	if err != nil {
		return nil, badRequest("invalid JSON: %v", err)
	}
	if v.Kind != laya.KindObject {
		return nil, badRequest("request body must be a JSON object")
	}
	r := &request{}
	haveCtx := false
	for _, f := range v.Obj {
		switch f.Key {
		case "context", "state":
			r.Context, haveCtx = f.Val, true
		case "spec", "questions":
			val := f.Val
			r.Spec = &val
		case "categories":
			val := f.Val
			r.Categories = &val
		case "model", "lang", "task":
			if f.Val.Kind == laya.KindNull {
				continue
			}
			if f.Val.Kind != laya.KindString {
				return nil, badRequest("field %q must be a string", f.Key)
			}
			switch f.Key {
			case "model":
				r.Route.Model = f.Val.Str
			case "lang":
				r.Route.Lang = f.Val.Str
			case "task":
				r.Route.Task = f.Val.Str
			}
		case "max_len":
			if f.Val.Kind == laya.KindNull {
				continue
			}
			n, convErr := strconv.Atoi(f.Val.Num)
			if f.Val.Kind != laya.KindNumber || convErr != nil || n < 0 || n > laya.MaxContext {
				return nil, badRequest("field \"max_len\" must be an integer between 1 and %d (tokens), e.g. 2048 or 4096", laya.MaxContext)
			}
			r.Route.MaxLen = n
		default:
			return nil, badRequest("unknown field %q (expected context, spec, model, lang, task, max_len; state/questions/categories are also accepted)", f.Key)
		}
	}
	if !haveCtx {
		return nil, badRequest("missing \"context\" (a string, object or array describing the state)")
	}
	return r, nil
}

func (s *server) predict(c fiber.Ctx) error {
	r, err := parseRequest(c)
	if err != nil {
		return err
	}
	if r.Spec == nil {
		return badRequest("missing \"spec\" (object of typed questions)")
	}
	spec, err := laya.ParseSpec(*r.Spec)
	if err != nil {
		return badRequest("%v", err)
	}
	res, err := s.router.Predict(r.Context, spec, r.Route)
	if err != nil {
		return predictError(err)
	}
	return c.JSON(res)
}

func predictError(err error) error {
	msg := err.Error()
	if strings.Contains(msg, "unknown model") || strings.Contains(msg, "exceed head_max_len") || strings.Contains(msg, "max_len") {
		return badRequest("%s", msg)
	}
	if strings.Contains(msg, "load ") {
		return &apiError{Status: fiber.StatusServiceUnavailable, Message: msg}
	}
	return err
}

func (s *server) route(c fiber.Ctx) error {
	r, err := parseRequest(c)
	if err != nil {
		return err
	}
	var spec laya.Spec
	if r.Spec != nil {
		if spec, err = laya.ParseSpec(*r.Spec); err != nil {
			return badRequest("%v", err)
		}
	}
	rt, err := s.router.Route(r.Context, spec, r.Route)
	if err != nil {
		return badRequest("%v", err)
	}
	return c.JSON(rt)
}

func (s *server) models(c fiber.Ctx) error {
	type info struct {
		Name    string `json:"name"`
		Repo    string `json:"repo,omitempty"`
		Encoder string `json:"encoder,omitempty"`
		MaxLen  int    `json:"max_len,omitempty"`
		Loaded  bool   `json:"loaded"`
	}
	loaded := map[string]bool{}
	for _, n := range s.router.Loaded() {
		loaded[n] = true
	}
	var out []info
	for _, n := range s.router.Available() {
		i := info{Name: n, Loaded: loaded[n]}
		if cfg, err := laya.LoadConfig(s.router.Dir(n)); err == nil {
			i.Repo, i.Encoder, i.MaxLen = cfg.Repo, cfg.Encoder, cfg.MaxLen
			if s.maxLen > 0 {
				i.MaxLen = s.maxLen
			}
		}
		out = append(out, i)
	}
	return c.JSON(fiber.Map{"models": out, "ready": s.ready.Load()})
}

func (s *server) listPresets(c fiber.Ctx) error {
	var out []fiber.Map
	for _, n := range laya.PresetNames() {
		out = append(out, fiber.Map{"name": n, "description": laya.PresetDescriptions[n], "endpoint": "/v1/presets/" + n})
	}
	return c.JSON(fiber.Map{"presets": out})
}

func (s *server) getPreset(c fiber.Ctx) error {
	v, err := laya.PresetSpecValue(c.Params("name"), nil)
	if err != nil {
		return &apiError{Status: fiber.StatusNotFound, Message: err.Error()}
	}
	return c.JSON(fiber.Map{"name": c.Params("name"), "spec": v})
}

func (s *server) runPreset(c fiber.Ctx) error {
	name := c.Params("name")
	r, err := parseRequest(c)
	if err != nil {
		return err
	}
	if r.Spec != nil {
		return badRequest("presets define their own spec; use /v1/predict to pass one")
	}
	spec, err := laya.Preset(name, r.Categories)
	if err != nil {
		return &apiError{Status: fiber.StatusNotFound, Message: err.Error()}
	}
	res, err := s.router.Predict(r.Context, spec, r.Route)
	if err != nil {
		return predictError(err)
	}
	return c.JSON(res)
}
