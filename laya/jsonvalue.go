package laya

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Value is an order-preserving JSON value.
//
// The upstream Python implementation serialises the state and structured criteria with
// json.dumps, whose output depends on dict insertion order. Go maps lose that order, so the
// request body is decoded into this structure and re-serialised with Python-compatible
// formatting (see PyDumps) so that token sequences match the reference implementation exactly.
type Value struct {
	Kind Kind
	Str  string  // KindString
	Num  string  // KindNumber: the literal as it appeared in the input
	Bool bool    // KindBool
	Arr  []Value // KindArray
	Obj  []Field // KindObject, in input order
}

// Field is one key/value pair of a JSON object.
type Field struct {
	Key string
	Val Value
}

// Kind is the JSON type of a Value.
type Kind uint8

const (
	KindNull Kind = iota
	KindBool
	KindNumber
	KindString
	KindArray
	KindObject
)

func (k Kind) String() string {
	switch k {
	case KindNull:
		return "null"
	case KindBool:
		return "bool"
	case KindNumber:
		return "number"
	case KindString:
		return "string"
	case KindArray:
		return "array"
	case KindObject:
		return "object"
	}
	return "?"
}

// StringValue builds a KindString Value.
func StringValue(s string) Value { return Value{Kind: KindString, Str: s} }

// NullValue is the JSON null.
var NullValue = Value{Kind: KindNull}

// Get returns the field named key of an object Value (and whether it exists).
func (v Value) Get(key string) (Value, bool) {
	if v.Kind != KindObject {
		return Value{}, false
	}
	for _, f := range v.Obj {
		if f.Key == key {
			return f.Val, true
		}
	}
	return Value{}, false
}

// IsNull reports whether v is JSON null (or the zero Value).
func (v Value) IsNull() bool { return v.Kind == KindNull }

// ParseValue decodes a JSON document into an order-preserving Value.
func ParseValue(data []byte) (Value, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return Value{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return Value{}, errors.New("trailing data after JSON value")
	}
	return v, nil
}

func decodeValue(dec *json.Decoder) (Value, error) {
	tok, err := dec.Token()
	if err != nil {
		return Value{}, err
	}
	return decodeFromToken(dec, tok)
}

func decodeFromToken(dec *json.Decoder, tok json.Token) (Value, error) {
	switch t := tok.(type) {
	case nil:
		return NullValue, nil
	case bool:
		return Value{Kind: KindBool, Bool: t}, nil
	case json.Number:
		return Value{Kind: KindNumber, Num: string(t)}, nil
	case string:
		return Value{Kind: KindString, Str: t}, nil
	case json.Delim:
		switch t {
		case '[':
			v := Value{Kind: KindArray, Arr: []Value{}}
			for dec.More() {
				e, err := decodeValue(dec)
				if err != nil {
					return Value{}, err
				}
				v.Arr = append(v.Arr, e)
			}
			if _, err := dec.Token(); err != nil { // ']'
				return Value{}, err
			}
			return v, nil
		case '{':
			v := Value{Kind: KindObject, Obj: []Field{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return Value{}, err
				}
				key, ok := kt.(string)
				if !ok {
					return Value{}, fmt.Errorf("object key is not a string: %v", kt)
				}
				e, err := decodeValue(dec)
				if err != nil {
					return Value{}, err
				}
				// Python dicts keep the first position of a duplicate key but the last value.
				replaced := false
				for i := range v.Obj {
					if v.Obj[i].Key == key {
						v.Obj[i].Val = e
						replaced = true
						break
					}
				}
				if !replaced {
					v.Obj = append(v.Obj, Field{Key: key, Val: e})
				}
			}
			if _, err := dec.Token(); err != nil { // '}'
				return Value{}, err
			}
			return v, nil
		}
	}
	return Value{}, fmt.Errorf("unexpected JSON token %v", tok)
}

// DumpOptions mirror the json.dumps arguments the upstream code uses.
type DumpOptions struct {
	EnsureASCII bool   // escape non-ASCII as \uXXXX (Python default True)
	ItemSep     string // default ", "
	KeySep      string // default ": "
}

// PyDumps serialises v the way Python's json.dumps would (default separators, sorted_keys=False).
func PyDumps(v Value, o DumpOptions) string {
	if o.ItemSep == "" {
		o.ItemSep = ", "
	}
	if o.KeySep == "" {
		o.KeySep = ": "
	}
	var b strings.Builder
	pyDump(&b, v, o)
	return b.String()
}

func pyDump(b *strings.Builder, v Value, o DumpOptions) {
	switch v.Kind {
	case KindNull:
		b.WriteString("null")
	case KindBool:
		if v.Bool {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case KindNumber:
		b.WriteString(pyNumber(v.Num))
	case KindString:
		pyQuote(b, v.Str, o.EnsureASCII)
	case KindArray:
		b.WriteByte('[')
		for i, e := range v.Arr {
			if i > 0 {
				b.WriteString(o.ItemSep)
			}
			pyDump(b, e, o)
		}
		b.WriteByte(']')
	case KindObject:
		b.WriteByte('{')
		for i, f := range v.Obj {
			if i > 0 {
				b.WriteString(o.ItemSep)
			}
			pyQuote(b, f.Key, o.EnsureASCII)
			b.WriteString(o.KeySep)
			pyDump(b, f.Val, o)
		}
		b.WriteByte('}')
	}
}

// pyQuote writes s as a Python json.dumps string literal.
func pyQuote(b *strings.Builder, s string, ensureASCII bool) {
	b.WriteByte('"')
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			switch c {
			case '"':
				b.WriteString(`\"`)
			case '\\':
				b.WriteString(`\\`)
			case '\n':
				b.WriteString(`\n`)
			case '\r':
				b.WriteString(`\r`)
			case '\t':
				b.WriteString(`\t`)
			case '\b':
				b.WriteString(`\b`)
			case '\f':
				b.WriteString(`\f`)
			default:
				if c < 0x20 || (c == 0x7f && ensureASCII) {
					fmt.Fprintf(b, `\u%04x`, c)
				} else {
					b.WriteByte(c)
				}
			}
			i++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			// invalid UTF-8: Python would have replaced it on decode; emit U+FFFD
			r = utf8.RuneError
		}
		if ensureASCII {
			if r >= 0x10000 {
				r -= 0x10000
				fmt.Fprintf(b, `\u%04x\u%04x`, 0xD800+(r>>10), 0xDC00+(r&0x3FF))
			} else {
				fmt.Fprintf(b, `\u%04x`, r)
			}
		} else {
			b.WriteRune(r)
		}
		i += size
	}
	b.WriteByte('"')
}

// pyNumber renders a JSON number literal as Python's json.dumps would after json.loads.
// Integers round-trip unchanged; floats take Python's repr() formatting.
func pyNumber(lit string) string {
	if !strings.ContainsAny(lit, ".eE") {
		if lit == "-0" {
			return "0"
		}
		return lit
	}
	f, err := strconv.ParseFloat(lit, 64)
	if err != nil && !math.IsInf(f, 0) { // out-of-range literals become ±Infinity, like json.loads
		return lit
	}
	return pyFloatRepr(f)
}

// pyFloatRepr mirrors float.__repr__: shortest round-trip digits, scientific notation when the
// decimal exponent is < -4 or >= 16, and a trailing ".0" for integral values.
func pyFloatRepr(f float64) string {
	if math.IsInf(f, 1) {
		return "Infinity"
	}
	if math.IsInf(f, -1) {
		return "-Infinity"
	}
	if math.IsNaN(f) {
		return "NaN"
	}
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // d.ddddde±XX
	mant, expStr, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expStr)
	neg := strings.HasPrefix(mant, "-")
	mant = strings.TrimPrefix(mant, "-")
	digits := strings.Replace(mant, ".", "", 1)
	var out string
	if exp < -4 || exp >= 16 {
		d := digits[:1]
		if len(digits) > 1 {
			d += "." + digits[1:]
		}
		sign := "+"
		if exp < 0 {
			sign = "-"
			exp = -exp
		}
		out = fmt.Sprintf("%se%s%02d", d, sign, exp)
	} else if exp >= 0 {
		if len(digits) <= exp+1 {
			out = digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
		} else {
			out = digits[:exp+1] + "." + digits[exp+1:]
		}
	} else {
		out = "0." + strings.Repeat("0", -exp-1) + digits
	}
	if neg {
		out = "-" + out
	}
	return out
}

// MarshalJSON renders the Value as standard JSON, preserving key order.
func (v Value) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	if err := writeJSON(&b, v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func writeJSON(b *bytes.Buffer, v Value) error {
	switch v.Kind {
	case KindNull:
		b.WriteString("null")
	case KindBool:
		if v.Bool {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case KindNumber:
		if v.Num == "" {
			b.WriteString("0")
		} else {
			b.WriteString(v.Num)
		}
	case KindString:
		enc, err := json.Marshal(v.Str)
		if err != nil {
			return err
		}
		b.Write(enc)
	case KindArray:
		b.WriteByte('[')
		for i, e := range v.Arr {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeJSON(b, e); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case KindObject:
		b.WriteByte('{')
		for i, f := range v.Obj {
			if i > 0 {
				b.WriteByte(',')
			}
			enc, err := json.Marshal(f.Key)
			if err != nil {
				return err
			}
			b.Write(enc)
			b.WriteByte(':')
			if err := writeJSON(b, f.Val); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	}
	return nil
}

// UnmarshalJSON decodes into an order-preserving Value.
func (v *Value) UnmarshalJSON(data []byte) error {
	parsed, err := ParseValue(data)
	if err != nil {
		return err
	}
	*v = parsed
	return nil
}

// FromGo converts ordinary Go data (as produced by encoding/json into any) into a Value.
// Map keys are emitted in sorted order, since Go maps carry no order.
func FromGo(x any) Value {
	switch t := x.(type) {
	case nil:
		return NullValue
	case bool:
		return Value{Kind: KindBool, Bool: t}
	case string:
		return StringValue(t)
	case float64:
		return Value{Kind: KindNumber, Num: strconv.FormatFloat(t, 'g', -1, 64)}
	case int:
		return Value{Kind: KindNumber, Num: strconv.Itoa(t)}
	case json.Number:
		return Value{Kind: KindNumber, Num: string(t)}
	case []any:
		v := Value{Kind: KindArray}
		for _, e := range t {
			v.Arr = append(v.Arr, FromGo(e))
		}
		return v
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sortStrings(keys)
		v := Value{Kind: KindObject}
		for _, k := range keys {
			v.Obj = append(v.Obj, Field{Key: k, Val: FromGo(t[k])})
		}
		return v
	case Value:
		return t
	}
	enc, err := json.Marshal(x)
	if err != nil {
		return StringValue(fmt.Sprint(x))
	}
	v, err := ParseValue(enc)
	if err != nil {
		return StringValue(fmt.Sprint(x))
	}
	return v
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
