// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package pretrained

import (
	"path/filepath"
	"regexp"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// WeightMapper maps weight names from pre-trained model format to GoMLX variable names.
type WeightMapper interface {
	// MapName converts a pre-trained weight name to a GoMLX variable scope/name.
	// Returns empty string if the weight should be skipped.
	MapName(pretrainedName string) string
}

// Loader handles loading pre-trained weights into a GoMLX context.
type Loader struct {
	// mapper converts weight names from source format to GoMLX format.
	mapper WeightMapper

	// strictMode when true, fails if any weights don't match.
	strictMode bool

	// loaded tracks which weights were loaded.
	loaded map[string]bool

	// skipped tracks which weights were skipped.
	skipped []string

	// mismatched tracks weights with shape mismatches.
	mismatched []string
}

// NewLoader creates a new pre-trained weight loader.
func NewLoader() *Loader {
	return &Loader{
		loaded:  make(map[string]bool),
		skipped: make([]string, 0),
	}
}

// WithMapper sets a custom weight name mapper.
func (l *Loader) WithMapper(mapper WeightMapper) *Loader {
	l.mapper = mapper
	return l
}

// WithStrictMode enables strict mode where all weights must match.
func (l *Loader) WithStrictMode(strict bool) *Loader {
	l.strictMode = strict
	return l
}

// LoadSafeTensors loads weights from a SafeTensors file into the context.
func (l *Loader) LoadSafeTensors(ctx *context.Context, filePath string) error {
	st, err := LoadSafeTensors(filePath)
	if err != nil {
		return err
	}
	defer st.Close()

	return l.loadFromTensorMap(ctx, st)
}

// LoadSafeTensorsDir loads weights from multiple SafeTensors files in a directory.
// This is common for large models that are sharded across multiple files.
func (l *Loader) LoadSafeTensorsDir(ctx *context.Context, dirPath string) error {
	files, err := filepath.Glob(filepath.Join(dirPath, "*.safetensors"))
	if err != nil {
		return errors.Wrapf(err, "failed to list SafeTensors files in %q", dirPath)
	}

	if len(files) == 0 {
		return errors.Errorf("no SafeTensors files found in %q", dirPath)
	}

	for _, filePath := range files {
		if err := l.LoadSafeTensors(ctx, filePath); err != nil {
			return errors.Wrapf(err, "failed to load %q", filePath)
		}
	}

	return nil
}

// TensorSource provides tensors by name (abstraction over SafeTensorsFile).
type TensorSource interface {
	TensorNames() []string
	LoadTensor(name string) (*tensors.Tensor, error)
}

// loadFromTensorMap loads weights from any TensorSource into the context.
func (l *Loader) loadFromTensorMap(ctx *context.Context, source TensorSource) error {
	for _, name := range source.TensorNames() {
		targetName := name
		if l.mapper != nil {
			targetName = l.mapper.MapName(name)
		}

		if targetName == "" {
			l.skipped = append(l.skipped, name)
			continue
		}

		tensor, err := source.LoadTensor(name)
		if err != nil {
			return errors.Wrapf(err, "failed to load tensor %q", name)
		}

		// Try to find and set the variable
		if err := l.setVariable(ctx, targetName, tensor); err != nil {
			if l.strictMode {
				return err
			}
			l.mismatched = append(l.mismatched, name+": "+err.Error())
		} else {
			l.loaded[name] = true
		}
	}

	return nil
}

// setVariable finds a variable by scope path and sets its value.
func (l *Loader) setVariable(ctx *context.Context, scopedName string, tensor *tensors.Tensor) error {
	// Parse the scoped name to navigate to the right context
	parts := strings.Split(scopedName, "/")
	if len(parts) == 0 {
		return errors.New("empty variable name")
	}

	// Navigate through scopes
	currentCtx := ctx
	for i := 0; i < len(parts)-1; i++ {
		currentCtx = currentCtx.In(parts[i])
	}

	varName := parts[len(parts)-1]

	// Look for the variable
	for v := range currentCtx.IterVariables() {
		// Check if this is the variable we're looking for
		vName := v.Name()
		// Variable names include full path, so check suffix
		if strings.HasSuffix(vName, "/"+varName) || vName == varName {
			// Check shape compatibility
			if !v.Shape().Equal(tensor.Shape()) {
				return errors.Errorf("shape mismatch for %q: expected %v, got %v",
					scopedName, v.Shape(), tensor.Shape())
			}
			if err := v.SetValue(tensor); err != nil {
				return errors.Wrapf(err, "failed to set value for %q", scopedName)
			}
			return nil
		}
	}

	// Variable not found - not necessarily an error if not in strict mode
	return errors.Errorf("variable %q not found in context", scopedName)
}

// LoadedCount returns the number of weights successfully loaded.
func (l *Loader) LoadedCount() int {
	return len(l.loaded)
}

// SkippedCount returns the number of weights skipped (no mapping).
func (l *Loader) SkippedCount() int {
	return len(l.skipped)
}

// MismatchedCount returns the number of weights with mismatches.
func (l *Loader) MismatchedCount() int {
	return len(l.mismatched)
}

// Skipped returns the list of skipped weight names.
func (l *Loader) Skipped() []string {
	return l.skipped
}

// Mismatched returns the list of mismatched weights with error descriptions.
func (l *Loader) Mismatched() []string {
	return l.mismatched
}

// IdentityMapper maps names without any transformation.
type IdentityMapper struct{}

// MapName returns the name unchanged.
func (m IdentityMapper) MapName(name string) string {
	return name
}

// PrefixMapper adds a prefix to all weight names.
type PrefixMapper struct {
	Prefix string
}

// MapName adds the configured prefix.
func (m PrefixMapper) MapName(name string) string {
	return m.Prefix + "/" + name
}

// RegexMapper uses regex patterns to transform weight names.
type RegexMapper struct {
	// Rules are applied in order. First matching rule is used.
	Rules []RegexRule
}

// RegexRule defines a regex-based name transformation.
type RegexRule struct {
	// Pattern is the regex pattern to match.
	Pattern *regexp.Regexp

	// Replacement is the replacement string (can use $1, $2, etc. for groups).
	Replacement string

	// Skip if true, matched weights are skipped entirely.
	Skip bool
}

// MapName applies regex rules to transform the name.
func (m *RegexMapper) MapName(name string) string {
	for _, rule := range m.Rules {
		if rule.Pattern.MatchString(name) {
			if rule.Skip {
				return ""
			}
			return rule.Pattern.ReplaceAllString(name, rule.Replacement)
		}
	}
	return name
}

// NewRegexMapper creates a RegexMapper from pattern/replacement pairs.
func NewRegexMapper(rules ...string) (*RegexMapper, error) {
	if len(rules)%2 != 0 {
		return nil, errors.New("rules must be pattern/replacement pairs")
	}

	mapper := &RegexMapper{
		Rules: make([]RegexRule, len(rules)/2),
	}

	for i := 0; i < len(rules); i += 2 {
		pattern, err := regexp.Compile(rules[i])
		if err != nil {
			return nil, errors.Wrapf(err, "invalid regex pattern %q", rules[i])
		}
		mapper.Rules[i/2] = RegexRule{
			Pattern:     pattern,
			Replacement: rules[i+1],
		}
	}

	return mapper, nil
}

// HuggingFaceBERTMapper provides common mappings for HuggingFace BERT models.
func HuggingFaceBERTMapper() *RegexMapper {
	return &RegexMapper{
		Rules: []RegexRule{
			// Skip pooler weights (often not needed for fine-tuning)
			{Pattern: regexp.MustCompile(`^pooler\..*`), Skip: true},
			// Embeddings
			{Pattern: regexp.MustCompile(`^embeddings\.word_embeddings\.weight$`),
				Replacement: "embeddings/word_embeddings/weights"},
			{Pattern: regexp.MustCompile(`^embeddings\.position_embeddings\.weight$`),
				Replacement: "embeddings/position_embeddings/weights"},
			{Pattern: regexp.MustCompile(`^embeddings\.token_type_embeddings\.weight$`),
				Replacement: "embeddings/token_type_embeddings/weights"},
			{Pattern: regexp.MustCompile(`^embeddings\.LayerNorm\.weight$`),
				Replacement: "embeddings/layer_norm/scale"},
			{Pattern: regexp.MustCompile(`^embeddings\.LayerNorm\.bias$`),
				Replacement: "embeddings/layer_norm/offset"},
			// Encoder layers
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.self\.query\.weight$`),
				Replacement: "encoder/layer_$1/attention/query/weights"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.self\.query\.bias$`),
				Replacement: "encoder/layer_$1/attention/query/bias"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.self\.key\.weight$`),
				Replacement: "encoder/layer_$1/attention/key/weights"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.self\.key\.bias$`),
				Replacement: "encoder/layer_$1/attention/key/bias"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.self\.value\.weight$`),
				Replacement: "encoder/layer_$1/attention/value/weights"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.self\.value\.bias$`),
				Replacement: "encoder/layer_$1/attention/value/bias"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.output\.dense\.weight$`),
				Replacement: "encoder/layer_$1/attention/output/weights"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.output\.dense\.bias$`),
				Replacement: "encoder/layer_$1/attention/output/bias"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.weight$`),
				Replacement: "encoder/layer_$1/attention/layer_norm/scale"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.bias$`),
				Replacement: "encoder/layer_$1/attention/layer_norm/offset"},
			// Feed-forward layers
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.intermediate\.dense\.weight$`),
				Replacement: "encoder/layer_$1/ffn/intermediate/weights"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.intermediate\.dense\.bias$`),
				Replacement: "encoder/layer_$1/ffn/intermediate/bias"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.output\.dense\.weight$`),
				Replacement: "encoder/layer_$1/ffn/output/weights"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.output\.dense\.bias$`),
				Replacement: "encoder/layer_$1/ffn/output/bias"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.output\.LayerNorm\.weight$`),
				Replacement: "encoder/layer_$1/ffn/layer_norm/scale"},
			{Pattern: regexp.MustCompile(`^encoder\.layer\.(\d+)\.output\.LayerNorm\.bias$`),
				Replacement: "encoder/layer_$1/ffn/layer_norm/offset"},
		},
	}
}
