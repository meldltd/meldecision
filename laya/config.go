package laya

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config is laya_config.json as written by export/export_onnx.py: the upstream
// rl_agent_config.json fields the runtime needs plus the tokenizer's special-token ids.
type Config struct {
	Name                 string             `json:"name"`
	Repo                 string             `json:"repo"`
	Encoder              string             `json:"encoder"`
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	Temperature          []float64          `json:"temperature"`
	TemperatureByOptions map[string]float64 `json:"temperature_by_options"`
	ActCosts             map[string]float64 `json:"act_costs"`
	SpecialTokens        SpecialTokens      `json:"special_tokens"`
	ONNX                 struct {
		File    string   `json:"file"`
		Opset   int      `json:"opset"`
		Inputs  []string `json:"inputs"`
		Outputs []string `json:"outputs"`
	} `json:"onnx"`
}

// SpecialTokens are the ids and surface forms of the tokenizer's special tokens.
type SpecialTokens struct {
	ClsID  int64  `json:"cls_id"`
	SepID  int64  `json:"sep_id"`
	MaskID int64  `json:"mask_id"`
	PadID  int64  `json:"pad_id"`
	Cls    string `json:"cls"`
	Sep    string `json:"sep"`
	Mask   string `json:"mask"`
	Pad    string `json:"pad"`
}

// MaxContext is the longest sequence any checkpoint accepts: both encoders (ModernBERT and
// mmBERT) are pre-trained with RoPE positions up to 8192 tokens and the exported ONNX graph
// has a dynamic sequence axis, so max_len may be raised up to this value at load time
// (Options.MaxLen) or per request (RouteRequest.MaxLen). The checkpoints were fine-tuned at
// 512/1024 tokens, so longer contexts are supported but not calibrated.
const MaxContext = 8192

// CheckMaxLen validates a max_len override for cfg: 0 means "use the checkpoint default";
// anything else must leave room for the question head and stay within MaxContext.
func (c *Config) CheckMaxLen(n int) error {
	switch {
	case n == 0:
		return nil
	case n < 0:
		return fmt.Errorf("max_len must be positive, got %d", n)
	case n > MaxContext:
		return fmt.Errorf("max_len %d exceeds the encoder limit of %d tokens", n, MaxContext)
	case n < c.HeadMaxLen+16:
		return fmt.Errorf("max_len %d is too small for head_max_len=%d (need at least %d)", n, c.HeadMaxLen, c.HeadMaxLen+16)
	}
	return nil
}

// withMaxLen returns c itself when n is 0 or equal to c.MaxLen, otherwise a shallow copy
// with MaxLen replaced (the slices and maps are shared read-only).
func (c *Config) withMaxLen(n int) *Config {
	if n == 0 || n == c.MaxLen {
		return c
	}
	cp := *c
	cp.MaxLen = n
	return &cp
}

// LoadConfig reads <dir>/laya_config.json and fills defaults.
func LoadConfig(dir string) (*Config, error) {
	p := filepath.Join(dir, "laya_config.json")
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w (run export/export_onnx.py first)", p, err)
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("parse %s: %w", p, err)
	}
	if c.MaxLen == 0 {
		c.MaxLen = 512
	}
	if c.HeadMaxLen == 0 {
		c.HeadMaxLen = 192
	}
	if len(c.Temperature) < 3 {
		c.Temperature = []float64{1, 1, 1}
	}
	if c.ONNX.File == "" {
		c.ONNX.File = "model.onnx"
	}
	if len(c.ONNX.Inputs) == 0 {
		c.ONNX.Inputs = []string{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
	}
	if len(c.ONNX.Outputs) == 0 {
		c.ONNX.Outputs = []string{"logits", "act_logits"}
	}
	if c.SpecialTokens.Mask == "" {
		c.SpecialTokens.Mask = "[MASK]"
	}
	return &c, nil
}

// temperatureFor mirrors upstream: temperature_by_options[temp_bucket] falling back to temperature[qtype].
func (c *Config) temperatureFor(t QType, k int) float64 {
	if v, ok := c.TemperatureByOptions[tempBucket(t, k)]; ok {
		return v
	}
	return c.Temperature[int(t)]
}

func tempBucket(t QType, k int) string {
	size := "11+"
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	}
	return t.String() + ":" + size
}
