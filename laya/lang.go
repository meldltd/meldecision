package laya

import (
	"regexp"
	"strings"
	"unicode"
)

// Port of upstream laya/lang.py: dependency-free script/language detection used to route
// between checkpoints. Script detection is exact; the Latin-script language guess is a
// stopword/diacritic heuristic and is best-effort by design.

type scriptRange struct {
	name   string
	ranges [][2]rune
}

var scriptRanges = []scriptRange{
	{"greek", [][2]rune{{0x0370, 0x03FF}, {0x1F00, 0x1FFF}}},
	{"cyrillic", [][2]rune{{0x0400, 0x052F}, {0x2DE0, 0x2DFF}, {0xA640, 0xA69F}}},
	{"hebrew", [][2]rune{{0x0590, 0x05FF}}},
	{"arabic", [][2]rune{{0x0600, 0x06FF}, {0x0750, 0x077F}, {0x08A0, 0x08FF}, {0xFB50, 0xFDFF}, {0xFE70, 0xFEFF}}},
	{"devanagari", [][2]rune{{0x0900, 0x097F}, {0xA8E0, 0xA8FF}}},
	{"bengali", [][2]rune{{0x0980, 0x09FF}}},
	{"gurmukhi", [][2]rune{{0x0A00, 0x0A7F}}},
	{"gujarati", [][2]rune{{0x0A80, 0x0AFF}}},
	{"oriya", [][2]rune{{0x0B00, 0x0B7F}}},
	{"tamil", [][2]rune{{0x0B80, 0x0BFF}}},
	{"telugu", [][2]rune{{0x0C00, 0x0C7F}}},
	{"kannada", [][2]rune{{0x0C80, 0x0CFF}}},
	{"malayalam", [][2]rune{{0x0D00, 0x0D7F}}},
	{"sinhala", [][2]rune{{0x0D80, 0x0DFF}}},
	{"thai", [][2]rune{{0x0E00, 0x0E7F}}},
	{"lao", [][2]rune{{0x0E80, 0x0EFF}}},
	{"tibetan", [][2]rune{{0x0F00, 0x0FFF}}},
	{"myanmar", [][2]rune{{0x1000, 0x109F}}},
	{"georgian", [][2]rune{{0x10A0, 0x10FF}}},
	{"ethiopic", [][2]rune{{0x1200, 0x137F}}},
	{"khmer", [][2]rune{{0x1780, 0x17FF}}},
	{"hangul", [][2]rune{{0x1100, 0x11FF}, {0x3130, 0x318F}, {0xAC00, 0xD7AF}}},
	{"kana", [][2]rune{{0x3040, 0x309F}, {0x30A0, 0x30FF}, {0x31F0, 0x31FF}}},
	{"han", [][2]rune{{0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}}},
}

var stopwords = map[string]map[string]bool{
	"en": set("the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
		"this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
		"will", "can", "would", "there", "their", "what", "which", "please", "we", "i"),
	"fr": set("le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
		"pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce"),
	"de": set("der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
		"auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber"),
	"es": set("el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
		"pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus"),
	"pt": set("os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
		"dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela"),
	"it": set("il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
		"questo", "questa", "anche", "come", "più", "sono", "nella", "alla"),
	"nl": set("het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
		"door", "maar", "ook", "worden", "deze", "naar", "wordt"),
}

var langOrder = []string{"en", "fr", "de", "es", "pt", "it", "nl"}

const nonEnDiacritics = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö"

// [^\W\d_]+ in Python: runs of letters (plus letter-like numerics).
var wordRe = regexp.MustCompile(`[\p{L}\p{Nl}\p{No}]+`)

func set(words ...string) map[string]bool {
	m := make(map[string]bool, len(words))
	for _, w := range words {
		m[w] = true
	}
	return m
}

// Detection is the upstream `analyse` result.
type Detection struct {
	Script           string             `json:"script"`
	ScriptProfile    map[string]float64 `json:"script_profile"`
	Language         *string            `json:"language"`
	IsEnglish        bool               `json:"is_english"`
	NonLatinFraction float64            `json:"non_latin_fraction"`
}

func iterText(v Value, depth int, out *[]string) {
	if depth > 6 {
		return
	}
	switch v.Kind {
	case KindString:
		*out = append(*out, v.Str)
	case KindObject:
		for _, f := range v.Obj {
			iterText(f.Val, depth+1, out)
		}
	case KindArray:
		for _, e := range v.Arr {
			iterText(e, depth+1, out)
		}
	}
}

// StateText flattens a state into the text used for detection (keys are ignored).
func StateText(state Value) string {
	var parts []string
	iterText(state, 0, &parts)
	s := strings.Join(parts, " ")
	r := []rune(s)
	if len(r) > 4000 {
		s = string(r[:4000])
	}
	return s
}

func isLatin(cp rune) bool { return cp < 0x0250 || (0x1E00 <= cp && cp <= 0x1EFF) }

func scriptOf(cp rune) string {
	for _, s := range scriptRanges {
		for _, r := range s.ranges {
			if r[0] <= cp && cp <= r[1] {
				return s.name
			}
		}
	}
	return ""
}

func scriptCounts(text string) (counts map[string]int, order []string, total int) {
	counts = map[string]int{}
	latin := 0
	for _, ch := range text {
		if !unicode.IsLetter(ch) {
			continue
		}
		if isLatin(ch) {
			latin++
			continue
		}
		if name := scriptOf(ch); name != "" {
			if _, seen := counts[name]; !seen {
				order = append(order, name)
			}
			counts[name]++
		}
	}
	// upstream inserts "latin" last, which matters for max() tie-breaking
	counts["latin"] = latin
	order = append(order, "latin")
	for _, v := range counts {
		total += v
	}
	return counts, order, total
}

// DetectScript returns the dominant script of text, or "unknown" if there are no letters.
// Ties resolve to the script first seen in the text, as Python's max() over dict items does.
func DetectScript(text string) string {
	counts, order, total := scriptCounts(text)
	if total == 0 {
		return "unknown"
	}
	best, bestN := "", -1
	for _, name := range order {
		if n := counts[name]; n > bestN {
			best, bestN = name, n
		}
	}
	return best
}

// ScriptProfile is the fraction of alphabetic characters per detected script.
func ScriptProfile(text string) map[string]float64 {
	counts, _, total := scriptCounts(text)
	out := map[string]float64{}
	if total == 0 {
		return out
	}
	for k, v := range counts {
		if v > 0 {
			out[k] = float64(v) / float64(total)
		}
	}
	return out
}

// GuessLatinLanguage is the best-effort language code for Latin-script text ("" when undecided).
func GuessLatinLanguage(text string) string {
	words := wordRe.FindAllString(text, -1)
	if len(words) < 4 {
		return ""
	}
	scores := map[string]int{}
	for _, w := range words {
		lw := strings.ToLower(w)
		for lg, sw := range stopwords {
			if sw[lw] {
				scores[lg]++
			}
		}
	}
	lowered := strings.ToLower(text)
	diac := 0
	for _, ch := range lowered {
		if strings.ContainsRune(nonEnDiacritics, ch) {
			diac++
		}
	}
	diacRate := float64(diac) / float64(max(1, len([]rune(lowered))))
	en := scores["en"]
	// Python's max() over the non-English languages returns the first entry ("fr") when all
	// scores are zero, which the diacritic rule below can still act on. Keep that behaviour.
	bestLg, best := "fr", 0
	for _, lg := range langOrder[1:] { // deterministic tie-break, like Python's dict order
		if s := scores[lg]; s > best {
			bestLg, best = lg, s
		}
	}
	enOrNone := func() string {
		if en > 0 {
			return "en"
		}
		return ""
	}
	if best == 0 && diacRate < 0.02 {
		return enOrNone()
	}
	if bestLg != "" && best >= max(2, en+2) {
		return bestLg
	}
	if diacRate >= 0.04 && bestLg != "" && best >= en {
		return bestLg
	}
	return enOrNone()
}

// Analyse is the full detection result for a state.
func Analyse(state Value) Detection {
	text := StateText(state)
	prof := ScriptProfile(text)
	script := DetectScript(text)
	nonLatin := 0.0
	if len(prof) > 0 {
		nonLatin = round4(1 - prof["latin"])
	}
	if script == "unknown" {
		return Detection{Script: "unknown", ScriptProfile: prof, IsEnglish: true}
	}
	if script != "latin" {
		return Detection{Script: script, ScriptProfile: prof, IsEnglish: false, NonLatinFraction: nonLatin}
	}
	lang := GuessLatinLanguage(text)
	d := Detection{Script: "latin", ScriptProfile: prof, IsEnglish: lang == "" || lang == "en", NonLatinFraction: nonLatin}
	if lang != "" {
		d.Language = &lang
	}
	return d
}

// IsEnglish reports whether the English checkpoint can be expected to read this state.
func IsEnglish(state Value) bool { return Analyse(state).IsEnglish }
