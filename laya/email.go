package laya

import (
	"regexp"
	"strings"
	"unicode"
)

// ws is Python's unicode-aware \s (RE2's \s is ASCII-only).
const ws = `[\s\p{Z}\x{85}\x{1c}-\x{1f}]`

// Port of upstream laya/email.py: strip quoted history, signatures and disclaimers so the
// model sees the message itself.

var quoteHeaders = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^` + ws + `*On .{0,300}wrote:` + ws + `*$`),
	regexp.MustCompile(`(?i)^` + ws + `*-{2,}` + ws + `*(Original|Forwarded) Message` + ws + `*-{2,}`),
	regexp.MustCompile(`^` + ws + `*_{8,}` + ws + `*$`),
	regexp.MustCompile(`(?i)^` + ws + `*From:` + ws + `.+$`),
}

var signatureMarkers = []*regexp.Regexp{
	regexp.MustCompile(`^` + ws + `*--` + ws + `*$`),
	regexp.MustCompile(`(?i)^` + ws + `*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\pL\pN_ ,!.]*$`),
	regexp.MustCompile(`(?i)^` + ws + `*sent from my (iphone|android|mobile|ipad)`),
}

var disclaimerRe = regexp.MustCompile(`(?i)(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|` +
	`if you (have )?received this (e-?mail|message) in error)`)

var (
	paragraphSplit = regexp.MustCompile(`\n` + ws + `*\n`)
	spaceRuns      = regexp.MustCompile(`[ \t]+`)
)

func anyMatch(res []*regexp.Regexp, s string) bool {
	for _, re := range res {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}

// CleanEmailBody removes quoted email history, signatures and disclaimers (max 3000 chars).
func CleanEmailBody(body string) string { return CleanEmailBodyN(body, 3000) }

// CleanEmailBodyN is CleanEmailBody with an explicit character limit.
func CleanEmailBodyN(body string, maxChars int) string {
	text := strings.NewReplacer("\r\n", "\n", "\r", "\n", `\n`, "\n").Replace(body)
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		if anyMatch(quoteHeaders, line) && len(lines) > 0 {
			break
		}
		if strings.HasPrefix(strings.TrimLeftFunc(line, unicode.IsSpace), ">") {
			continue
		}
		lines = append(lines, strings.TrimRightFunc(line, unicode.IsSpace))
	}
	cut := len(lines)
	start := max(1, min(int(float64(len(lines))*0.6), len(lines)-8))
	for i := start; i < len(lines); i++ {
		if len(strings.TrimSpace(lines[i])) <= 40 && anyMatch(signatureMarkers, lines[i]) {
			cut = i
			break
		}
	}
	lines = lines[:cut]
	var paras []string
	for _, p := range paragraphSplit.Split(strings.Join(lines, "\n"), -1) {
		if disclaimerRe.MatchString(p) {
			continue
		}
		if t := strings.TrimSpace(p); t != "" {
			paras = append(paras, t)
		}
	}
	out := spaceRuns.ReplaceAllString(strings.Join(paras, "\n\n"), " ")
	if r := []rune(out); len(r) > maxChars {
		out = string(r[:maxChars])
	}
	return out
}

// EmailState builds the state object for email classification: {"subject", "body", ["from"], extra...}.
func EmailState(subject, body, sender string, clean bool, extra []Field) Value {
	b := body
	if clean {
		b = CleanEmailBody(body)
	}
	v := Value{Kind: KindObject, Obj: []Field{
		{Key: "subject", Val: StringValue(strings.TrimSpace(subject))},
		{Key: "body", Val: StringValue(b)},
	}}
	if sender != "" {
		v.Obj = append(v.Obj, Field{Key: "from", Val: StringValue(sender)})
	}
	for _, f := range extra {
		if !f.Val.IsNull() {
			v.Obj = append(v.Obj, f)
		}
	}
	return v
}
