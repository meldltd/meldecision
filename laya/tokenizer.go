package laya

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/daulet/tokenizers"
)

// Tokenizer wraps the Hugging Face tokenizers library (Rust, via cgo) so that the Go runtime
// tokenizes byte-for-byte like the Python reference.
type Tokenizer struct {
	tk *tokenizers.Tokenizer
}

// LoadTokenizer reads <dir>/tokenizer.json.
func LoadTokenizer(dir string) (*Tokenizer, error) {
	p := filepath.Join(dir, "tokenizer.json")
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", p, err)
	}
	// Default options: special tokens appearing in the text are recognised, like the Python
	// tokenizer with split_special_tokens=False. build_sequence strips [MASK] beforehand.
	tk, err := tokenizers.FromBytes(data)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer %s: %w", p, err)
	}
	return &Tokenizer{tk: tk}, nil
}

// Encode returns token ids without special tokens (tok(text, add_special_tokens=False)).
func (t *Tokenizer) Encode(text string) []int64 {
	ids, _ := t.tk.Encode(text, false)
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[i] = int64(id)
	}
	return out
}

// Close releases the native tokenizer.
func (t *Tokenizer) Close() error {
	if t == nil || t.tk == nil {
		return nil
	}
	return t.tk.Close()
}
