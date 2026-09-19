package laya

import (
	"strings"
	"testing"
)

func TestCheckMaxLen(t *testing.T) {
	cfg := &Config{MaxLen: 1024, HeadMaxLen: 256}
	for _, n := range []int{0, 512, 1024, 2048, 4096, MaxContext} {
		if err := cfg.CheckMaxLen(n); err != nil {
			t.Errorf("CheckMaxLen(%d): unexpected error %v", n, err)
		}
	}
	for n, want := range map[int]string{-1: "positive", 100: "too small", MaxContext + 1: "exceeds"} {
		err := cfg.CheckMaxLen(n)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("CheckMaxLen(%d) = %v, want error containing %q", n, err, want)
		}
	}
	if c := cfg.withMaxLen(0); c != cfg {
		t.Error("withMaxLen(0) should return the same config")
	}
	if c := cfg.withMaxLen(1024); c != cfg {
		t.Error("withMaxLen(same) should return the same config")
	}
	c := cfg.withMaxLen(4096)
	if c == cfg || c.MaxLen != 4096 || cfg.MaxLen != 1024 || c.HeadMaxLen != 256 {
		t.Errorf("withMaxLen(4096) = %+v (orig %+v)", c, cfg)
	}
}
