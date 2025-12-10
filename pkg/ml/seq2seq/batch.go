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

package seq2seq

import (
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// Batch represents a batch of inputs being processed through a seq2seq model.
// It maintains state across encoder and decoder calls, including:
//   - Input tensors (input IDs and attention mask)
//   - Encoder hidden states
//   - Decoder KV cache for efficient autoregressive generation
//   - Generated output tokens
type Batch struct {
	model *Model

	// Input tensors.
	InputIDs      *tensors.Tensor // Shape: [batch_size, encoder_seq_len]
	AttentionMask *tensors.Tensor // Shape: [batch_size, encoder_seq_len], optional

	// Batch dimensions.
	batchSize     int
	encoderSeqLen int

	// Encoder outputs - persisted across decoder calls.
	EncoderHiddenStates *tensors.Tensor   // Shape: [batch_size, encoder_seq_len, hidden_size]
	EncoderOutputs      []*tensors.Tensor // All encoder outputs if multiple

	// Decoder KV cache - updated at each generation step.
	// For each layer, contains key and value tensors.
	// Keys shape:   [batch_size, num_heads, seq_len, head_dim]
	// Values shape: [batch_size, num_heads, seq_len, head_dim]
	kvCache *KVCache

	// Generation state.
	decoderInputIDs *tensors.Tensor // Current decoder input IDs
	generatedIDs    [][]int32       // Generated token IDs per batch item
	currentPosition int             // Current position in generation

	// Flags.
	encoderRun    bool // Whether encoder has been run
	decoderInitd  bool // Whether decoder has been initialized
	generationDone bool // Whether generation is complete
}

// KVCache manages the key-value cache for decoder self-attention and cross-attention.
type KVCache struct {
	// NumLayers is the number of decoder layers.
	NumLayers int

	// SelfAttentionKeys holds self-attention key tensors for each layer.
	// Shape per layer: [batch_size, num_heads, seq_len, head_dim]
	SelfAttentionKeys []*tensors.Tensor

	// SelfAttentionValues holds self-attention value tensors for each layer.
	// Shape per layer: [batch_size, num_heads, seq_len, head_dim]
	SelfAttentionValues []*tensors.Tensor

	// CrossAttentionKeys holds cross-attention key tensors for each layer.
	// These are computed once from encoder outputs and reused.
	// Shape per layer: [batch_size, num_heads, encoder_seq_len, head_dim]
	CrossAttentionKeys []*tensors.Tensor

	// CrossAttentionValues holds cross-attention value tensors for each layer.
	// Shape per layer: [batch_size, num_heads, encoder_seq_len, head_dim]
	CrossAttentionValues []*tensors.Tensor

	// CurrentLength tracks the current sequence length in the self-attention cache.
	CurrentLength int
}

// NewKVCache creates a new KVCache for the given number of layers.
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		NumLayers:            numLayers,
		SelfAttentionKeys:    make([]*tensors.Tensor, numLayers),
		SelfAttentionValues:  make([]*tensors.Tensor, numLayers),
		CrossAttentionKeys:   make([]*tensors.Tensor, numLayers),
		CrossAttentionValues: make([]*tensors.Tensor, numLayers),
		CurrentLength:        0,
	}
}

// SetSelfAttention sets the self-attention KV cache for a specific layer.
func (kv *KVCache) SetSelfAttention(layer int, keys, values *tensors.Tensor) {
	kv.SelfAttentionKeys[layer] = keys
	kv.SelfAttentionValues[layer] = values
}

// SetCrossAttention sets the cross-attention KV cache for a specific layer.
func (kv *KVCache) SetCrossAttention(layer int, keys, values *tensors.Tensor) {
	kv.CrossAttentionKeys[layer] = keys
	kv.CrossAttentionValues[layer] = values
}

// GetSelfAttentionKV returns the self-attention key-value tensors for all layers.
// Returns two slices: keys and values, each with NumLayers elements.
func (kv *KVCache) GetSelfAttentionKV() ([]*tensors.Tensor, []*tensors.Tensor) {
	return kv.SelfAttentionKeys, kv.SelfAttentionValues
}

// GetCrossAttentionKV returns the cross-attention key-value tensors for all layers.
func (kv *KVCache) GetCrossAttentionKV() ([]*tensors.Tensor, []*tensors.Tensor) {
	return kv.CrossAttentionKeys, kv.CrossAttentionValues
}

// Finalize releases all tensors in the KV cache.
func (kv *KVCache) Finalize() {
	for i := range kv.SelfAttentionKeys {
		if kv.SelfAttentionKeys[i] != nil {
			kv.SelfAttentionKeys[i].FinalizeAll()
			kv.SelfAttentionKeys[i] = nil
		}
		if kv.SelfAttentionValues[i] != nil {
			kv.SelfAttentionValues[i].FinalizeAll()
			kv.SelfAttentionValues[i] = nil
		}
		if kv.CrossAttentionKeys[i] != nil {
			kv.CrossAttentionKeys[i].FinalizeAll()
			kv.CrossAttentionKeys[i] = nil
		}
		if kv.CrossAttentionValues[i] != nil {
			kv.CrossAttentionValues[i].FinalizeAll()
			kv.CrossAttentionValues[i] = nil
		}
	}
}

// BatchSize returns the batch size.
func (b *Batch) BatchSize() int {
	return b.batchSize
}

// EncoderSeqLen returns the encoder sequence length.
func (b *Batch) EncoderSeqLen() int {
	return b.encoderSeqLen
}

// EncoderRun returns whether the encoder has been run.
func (b *Batch) EncoderRun() bool {
	return b.encoderRun
}

// DecoderInitialized returns whether the decoder has been initialized.
func (b *Batch) DecoderInitialized() bool {
	return b.decoderInitd
}

// GenerationDone returns whether generation is complete.
func (b *Batch) GenerationDone() bool {
	return b.generationDone
}

// GetGeneratedIDs returns the generated token IDs for all batch items.
func (b *Batch) GetGeneratedIDs() [][]int32 {
	return b.generatedIDs
}

// GetKVCache returns the current KV cache.
func (b *Batch) GetKVCache() *KVCache {
	return b.kvCache
}

// InitKVCache initializes the KV cache with the given number of layers.
func (b *Batch) InitKVCache(numLayers int) {
	b.kvCache = NewKVCache(numLayers)
}

// RunEncoder executes the encoder model on the input tensors.
// The encoder hidden states are stored in the batch for use by the decoder.
func (b *Batch) RunEncoder() error {
	if b.encoderRun {
		return errors.New("encoder has already been run for this batch")
	}

	model := b.model
	if model.EncoderModel == nil {
		return errors.New("encoder model is not configured")
	}

	// Dispatch to appropriate backend.
	switch model.backendType {
	case BackendGoMLX, BackendXLA:
		return b.runEncoderGoMLX()
	case BackendORT:
		return errors.New("ORT backend not implemented yet")
	default:
		return errors.Errorf("unknown backend type: %s", model.backendType)
	}
}

// RunDecoderInit runs the decoder initialization step.
// This is called once after the encoder to set up initial KV cache from encoder outputs.
func (b *Batch) RunDecoderInit() error {
	if !b.encoderRun {
		return errors.New("encoder must be run before decoder init")
	}
	if b.decoderInitd {
		return errors.New("decoder init has already been run for this batch")
	}

	model := b.model
	if model.DecoderInitModel == nil {
		// Some models don't have a separate decoder-init, skip.
		b.decoderInitd = true
		return nil
	}

	switch model.backendType {
	case BackendGoMLX, BackendXLA:
		return b.runDecoderInitGoMLX()
	case BackendORT:
		return errors.New("ORT backend not implemented yet")
	default:
		return errors.Errorf("unknown backend type: %s", model.backendType)
	}
}

// RunDecoderStep runs a single decoder step with the current input and KV cache.
// Returns the logits for the next token prediction.
func (b *Batch) RunDecoderStep(inputIDs *tensors.Tensor) (*tensors.Tensor, error) {
	if !b.encoderRun {
		return nil, errors.New("encoder must be run before decoder step")
	}

	model := b.model
	if model.DecoderModel == nil && model.DecoderInitModel == nil {
		return nil, errors.New("decoder model is not configured")
	}

	switch model.backendType {
	case BackendGoMLX, BackendXLA:
		return b.runDecoderStepGoMLX(inputIDs)
	case BackendORT:
		return nil, errors.New("ORT backend not implemented yet")
	default:
		return nil, errors.Errorf("unknown backend type: %s", model.backendType)
	}
}

// Destroy releases all resources held by the batch.
func (b *Batch) Destroy() {
	if b.EncoderHiddenStates != nil {
		b.EncoderHiddenStates.FinalizeAll()
		b.EncoderHiddenStates = nil
	}
	for _, t := range b.EncoderOutputs {
		if t != nil {
			t.FinalizeAll()
		}
	}
	b.EncoderOutputs = nil

	if b.kvCache != nil {
		b.kvCache.Finalize()
		b.kvCache = nil
	}

	if b.decoderInputIDs != nil {
		b.decoderInputIDs.FinalizeAll()
		b.decoderInputIDs = nil
	}

	// Note: InputIDs and AttentionMask are owned by the caller, not finalized here.
}

// DestroyDecoder releases only decoder-related resources, keeping encoder outputs.
// Useful when running multiple generations from the same encoded input.
func (b *Batch) DestroyDecoder() {
	if b.kvCache != nil {
		b.kvCache.Finalize()
		b.kvCache = nil
	}

	if b.decoderInputIDs != nil {
		b.decoderInputIDs.FinalizeAll()
		b.decoderInputIDs = nil
	}

	b.decoderInitd = false
	b.generationDone = false
	b.generatedIDs = nil
	b.currentPosition = 0
}
