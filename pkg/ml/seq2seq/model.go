/*
 *	Copyright 2024 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// Package seq2seq provides infrastructure for running sequence-to-sequence models
// like T5, BART, and other encoder-decoder architectures using GoMLX.
//
// This package supports:
//   - Loading and running ONNX-exported seq2seq models
//   - Encoder execution with hidden state extraction
//   - Decoder execution with KV cache management
//   - Greedy and sampling-based generation
//
// Example usage:
//
//	model := seq2seq.NewModel(backend)
//	model.LoadEncoder("encoder.onnx")
//	model.LoadDecoderInit("decoder_init.onnx")
//	model.LoadDecoder("decoder.onnx")
//
//	batch := model.NewBatch(inputIDs, attentionMask)
//	batch.RunEncoder()
//	output := batch.GenerateGreedy(maxLength)
package seq2seq

import (
	"fmt"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// BackendType represents the backend to use for execution.
type BackendType string

const (
	// BackendGoMLX uses GoMLX/XLA backend for execution.
	BackendGoMLX BackendType = "GO"

	// BackendXLA is an alias for BackendGoMLX.
	BackendXLA BackendType = "XLA"

	// BackendORT uses ONNX Runtime backend for execution (when available).
	BackendORT BackendType = "ORT"
)

// Model represents a seq2seq model with separate encoder and decoder components.
// It manages the model loading, execution context, and provides the interface
// for creating batches for inference.
type Model struct {
	mu sync.RWMutex

	// Backend for computation.
	backend backends.Backend

	// BackendType specifies which execution backend to use.
	backendType BackendType

	// Context holds model parameters and variables.
	ctx *context.Context

	// EncoderModel holds the encoder graph function and executor.
	// This is optional - some seq2seq models may use a combined encoder-decoder.
	EncoderModel *SubModel

	// DecoderInitModel holds the decoder initialization graph.
	// This is called once per batch with encoder outputs to initialize KV cache.
	DecoderInitModel *SubModel

	// DecoderModel holds the main decoder graph with KV cache support.
	// This is called iteratively during generation.
	DecoderModel *SubModel

	// Config holds model configuration parameters.
	Config *ModelConfig
}

// SubModel represents a component model (encoder, decoder-init, or decoder).
type SubModel struct {
	// Name identifies this component (e.g., "encoder", "decoder_init", "decoder").
	Name string

	// GraphFn is the graph-building function for this component.
	GraphFn any

	// Exec is the compiled executor for this component.
	Exec *context.Exec

	// InputNames lists the expected input tensor names in order.
	InputNames []string

	// OutputNames lists the output tensor names in order.
	OutputNames []string

	// InputShapes stores the shapes of expected inputs.
	InputShapes []shapes.Shape

	// OutputShapes stores the shapes of outputs.
	OutputShapes []shapes.Shape
}

// ModelConfig holds configuration for the seq2seq model.
type ModelConfig struct {
	// VocabSize is the size of the vocabulary.
	VocabSize int

	// HiddenSize is the dimension of hidden states.
	HiddenSize int

	// NumHeads is the number of attention heads.
	NumHeads int

	// NumLayers is the number of encoder/decoder layers.
	NumLayers int

	// HeadDim is the dimension per attention head (typically HiddenSize/NumHeads).
	HeadDim int

	// MaxLength is the maximum sequence length.
	MaxLength int

	// DType is the data type for computations.
	DType dtypes.DType

	// PadTokenID is the token ID used for padding.
	PadTokenID int32

	// EOSTokenID is the end-of-sequence token ID.
	EOSTokenID int32

	// DecoderStartTokenID is the token ID to start decoder generation.
	DecoderStartTokenID int32

	// ForcedBOSTokenID, if set, forces this token as the first generated token.
	ForcedBOSTokenID int32

	// KVCacheDType specifies the dtype for KV cache (may differ from model dtype).
	KVCacheDType dtypes.DType
}

// DefaultModelConfig returns a ModelConfig with common default values.
func DefaultModelConfig() *ModelConfig {
	return &ModelConfig{
		DType:               dtypes.Float32,
		KVCacheDType:        dtypes.Float32,
		MaxLength:           512,
		PadTokenID:          0,
		EOSTokenID:          1,
		DecoderStartTokenID: 0,
		ForcedBOSTokenID:    -1, // Disabled by default
	}
}

// NewModel creates a new seq2seq Model with the given backend.
func NewModel(backend backends.Backend) *Model {
	return &Model{
		backend:     backend,
		backendType: BackendGoMLX,
		ctx:         context.New(),
		Config:      DefaultModelConfig(),
	}
}

// WithBackendType sets the backend type to use for execution.
func (m *Model) WithBackendType(bt BackendType) *Model {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.backendType = bt
	return m
}

// WithContext sets a custom context for the model.
func (m *Model) WithContext(ctx *context.Context) *Model {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ctx = ctx
	return m
}

// WithConfig sets the model configuration.
func (m *Model) WithConfig(cfg *ModelConfig) *Model {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Config = cfg
	return m
}

// Backend returns the backend used by this model.
func (m *Model) Backend() backends.Backend {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.backend
}

// Context returns the context used by this model.
func (m *Model) Context() *context.Context {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.ctx
}

// SetEncoder sets the encoder submodel.
func (m *Model) SetEncoder(encoder *SubModel) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.EncoderModel = encoder
	m.EncoderModel.Name = "encoder"
}

// SetDecoderInit sets the decoder initialization submodel.
func (m *Model) SetDecoderInit(decoderInit *SubModel) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.DecoderInitModel = decoderInit
	m.DecoderInitModel.Name = "decoder_init"
}

// SetDecoder sets the main decoder submodel.
func (m *Model) SetDecoder(decoder *SubModel) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.DecoderModel = decoder
	m.DecoderModel.Name = "decoder"
}

// HasEncoder returns true if an encoder model is configured.
func (m *Model) HasEncoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.EncoderModel != nil
}

// HasDecoderInit returns true if a decoder-init model is configured.
func (m *Model) HasDecoderInit() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.DecoderInitModel != nil
}

// HasDecoder returns true if a decoder model is configured.
func (m *Model) HasDecoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.DecoderModel != nil
}

// Validate checks that the model is properly configured.
func (m *Model) Validate() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.backend == nil {
		return errors.New("model backend is not set")
	}
	if m.ctx == nil {
		return errors.New("model context is not set")
	}
	if m.Config == nil {
		return errors.New("model config is not set")
	}

	// For seq2seq, we need at least an encoder and decoder.
	if m.EncoderModel == nil {
		return errors.New("encoder model is not set")
	}
	if m.DecoderModel == nil && m.DecoderInitModel == nil {
		return errors.New("at least one of decoder or decoder-init model must be set")
	}

	return nil
}

// NewBatch creates a new Batch for processing inputs through the seq2seq model.
func (m *Model) NewBatch(inputIDs, attentionMask *tensors.Tensor) (*Batch, error) {
	if err := m.Validate(); err != nil {
		return nil, errors.WithMessage(err, "model validation failed")
	}

	if inputIDs == nil {
		return nil, errors.New("inputIDs cannot be nil")
	}

	batch := &Batch{
		model:         m,
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	}

	// Infer batch size and sequence length from input shape.
	inputShape := inputIDs.Shape()
	if inputShape.Rank() != 2 {
		return nil, errors.Errorf("inputIDs must be 2D [batch, seq_len], got shape %s", inputShape)
	}
	batch.batchSize = inputShape.Dimensions[0]
	batch.encoderSeqLen = inputShape.Dimensions[1]

	return batch, nil
}

// String returns a human-readable description of the model.
func (m *Model) String() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	hasEncoder := m.EncoderModel != nil
	hasDecoderInit := m.DecoderInitModel != nil
	hasDecoder := m.DecoderModel != nil

	return fmt.Sprintf("Seq2SeqModel{backend=%s, hasEncoder=%t, hasDecoderInit=%t, hasDecoder=%t}",
		m.backendType, hasEncoder, hasDecoderInit, hasDecoder)
}
