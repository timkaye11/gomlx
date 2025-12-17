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
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// runEncoderGoMLX executes the encoder using GoMLX backend.
func (b *Batch) runEncoderGoMLX() error {
	model := b.model
	encoder := model.EncoderModel

	if encoder.Exec == nil {
		return errors.New("encoder executor not initialized")
	}

	// Prepare inputs.
	var inputs []any
	inputs = append(inputs, b.InputIDs)
	if b.AttentionMask != nil {
		inputs = append(inputs, b.AttentionMask)
	}

	// Execute encoder.
	outputs, err := encoder.Exec.Exec(inputs...)
	if err != nil {
		return errors.WithMessage(err, "encoder execution failed")
	}

	if len(outputs) == 0 {
		return errors.New("encoder returned no outputs")
	}

	// Store encoder outputs.
	// The first output is typically the hidden states.
	b.EncoderHiddenStates = outputs[0]
	b.EncoderOutputs = outputs
	b.encoderRun = true

	return nil
}

// runDecoderInitGoMLX executes the decoder initialization using GoMLX backend.
func (b *Batch) runDecoderInitGoMLX() error {
	model := b.model
	decoderInit := model.DecoderInitModel

	if decoderInit == nil {
		// No decoder-init model, just mark as initialized.
		b.decoderInitd = true
		return nil
	}

	if decoderInit.Exec == nil {
		return errors.New("decoder-init executor not initialized")
	}

	// Create initial decoder input (start token).
	startTokenID := model.Config.DecoderStartTokenID
	decoderStartIDs := make([]int32, b.batchSize)
	for i := range decoderStartIDs {
		decoderStartIDs[i] = startTokenID
	}
	b.decoderInputIDs = tensors.FromFlatDataAndDimensions(decoderStartIDs, b.batchSize, 1)

	// Prepare inputs: encoder hidden states and initial decoder input.
	var inputs []any
	inputs = append(inputs, b.EncoderHiddenStates)
	if b.AttentionMask != nil {
		inputs = append(inputs, b.AttentionMask)
	}
	inputs = append(inputs, b.decoderInputIDs)

	// Execute decoder-init.
	outputs, err := decoderInit.Exec.Exec(inputs...)
	if err != nil {
		return errors.WithMessage(err, "decoder-init execution failed")
	}

	// Parse outputs - first is logits, rest are KV cache tensors.
	if len(outputs) < 1 {
		return errors.New("decoder-init returned no outputs")
	}

	// Initialize KV cache from outputs.
	numLayers := model.Config.NumLayers
	if numLayers == 0 {
		// Try to infer from outputs: (logits, then pairs of k/v for each layer)
		// Assuming output structure: logits, (self_k, self_v, cross_k, cross_v) * numLayers
		// Or: logits, (self_k, self_v) * numLayers (if cross-attention is pre-computed)
		numLayers = (len(outputs) - 1) / 2 // Conservative estimate
	}

	b.InitKVCache(numLayers)

	// Store KV cache tensors from decoder-init output.
	// The exact structure depends on the model, but typically:
	// Output[0] = logits
	// Output[1:] = KV cache tensors
	err = b.parseKVCacheFromOutputs(outputs[1:])
	if err != nil {
		return errors.WithMessage(err, "failed to parse KV cache from decoder-init outputs")
	}

	b.decoderInitd = true
	b.currentPosition = 1

	// Initialize generated IDs storage.
	b.generatedIDs = make([][]int32, b.batchSize)
	for i := range b.generatedIDs {
		b.generatedIDs[i] = make([]int32, 0, model.Config.MaxLength)
	}

	return nil
}

// runDecoderStepGoMLX executes a single decoder step using GoMLX backend.
func (b *Batch) runDecoderStepGoMLX(inputIDs *tensors.Tensor) (*tensors.Tensor, error) {
	model := b.model
	decoder := model.DecoderModel

	// Use decoder-init if no separate decoder.
	if decoder == nil {
		decoder = model.DecoderInitModel
	}

	if decoder == nil || decoder.Exec == nil {
		return nil, errors.New("decoder executor not initialized")
	}

	// Prepare inputs.
	var inputs []any

	// Add encoder hidden states.
	inputs = append(inputs, b.EncoderHiddenStates)

	// Add encoder attention mask if present.
	if b.AttentionMask != nil {
		inputs = append(inputs, b.AttentionMask)
	}

	// Add decoder input IDs.
	inputs = append(inputs, inputIDs)

	// Add KV cache if present.
	if b.kvCache != nil {
		for i := 0; i < b.kvCache.NumLayers; i++ {
			if b.kvCache.SelfAttentionKeys[i] != nil {
				inputs = append(inputs, b.kvCache.SelfAttentionKeys[i])
				inputs = append(inputs, b.kvCache.SelfAttentionValues[i])
			}
		}
	}

	// Execute decoder step.
	outputs, err := decoder.Exec.Exec(inputs...)
	if err != nil {
		return nil, errors.WithMessage(err, "decoder step execution failed")
	}

	if len(outputs) < 1 {
		return nil, errors.New("decoder step returned no outputs")
	}

	// First output is logits.
	logits := outputs[0]

	// Update KV cache with new values from outputs.
	if len(outputs) > 1 {
		err = b.updateKVCacheFromOutputs(outputs[1:])
		if err != nil {
			return nil, errors.WithMessage(err, "failed to update KV cache")
		}
	}

	b.currentPosition++

	return logits, nil
}

// parseKVCacheFromOutputs parses KV cache tensors from model outputs.
// Expected format: (self_k, self_v) pairs for each layer,
// optionally followed by (cross_k, cross_v) pairs.
func (b *Batch) parseKVCacheFromOutputs(outputs []*tensors.Tensor) error {
	if b.kvCache == nil {
		return errors.New("KV cache not initialized")
	}

	numLayers := b.kvCache.NumLayers
	expectedMinOutputs := numLayers * 2 // At minimum: self-attention k/v pairs

	if len(outputs) < expectedMinOutputs {
		return errors.Errorf("expected at least %d KV cache outputs, got %d",
			expectedMinOutputs, len(outputs))
	}

	// Parse self-attention KV pairs.
	for i := 0; i < numLayers; i++ {
		keyIdx := i * 2
		valueIdx := i*2 + 1
		if keyIdx >= len(outputs) || valueIdx >= len(outputs) {
			break
		}
		b.kvCache.SetSelfAttention(i, outputs[keyIdx], outputs[valueIdx])
	}

	// If there are more outputs, they're cross-attention KV pairs.
	crossStartIdx := numLayers * 2
	if len(outputs) >= crossStartIdx+numLayers*2 {
		for i := 0; i < numLayers; i++ {
			keyIdx := crossStartIdx + i*2
			valueIdx := crossStartIdx + i*2 + 1
			if keyIdx >= len(outputs) || valueIdx >= len(outputs) {
				break
			}
			b.kvCache.SetCrossAttention(i, outputs[keyIdx], outputs[valueIdx])
		}
	}

	return nil
}

// updateKVCacheFromOutputs updates the KV cache with new values.
// This is called after each decoder step to extend the cache.
func (b *Batch) updateKVCacheFromOutputs(outputs []*tensors.Tensor) error {
	if b.kvCache == nil {
		return errors.New("KV cache not initialized")
	}

	numLayers := b.kvCache.NumLayers

	// Update self-attention KV pairs.
	// The new outputs replace the old cache (they contain the full sequence).
	for i := 0; i < numLayers; i++ {
		keyIdx := i * 2
		valueIdx := i*2 + 1
		if keyIdx >= len(outputs) || valueIdx >= len(outputs) {
			break
		}

		// Finalize old tensors before replacing.
		if b.kvCache.SelfAttentionKeys[i] != nil {
			b.kvCache.SelfAttentionKeys[i].FinalizeAll()
		}
		if b.kvCache.SelfAttentionValues[i] != nil {
			b.kvCache.SelfAttentionValues[i].FinalizeAll()
		}

		b.kvCache.SelfAttentionKeys[i] = outputs[keyIdx]
		b.kvCache.SelfAttentionValues[i] = outputs[valueIdx]
	}

	b.kvCache.CurrentLength = b.currentPosition + 1

	return nil
}

// CreateEncoderExec creates an executor for the encoder graph function.
func CreateEncoderExec(model *Model, graphFn func(ctx *context.Context, inputIDs, attentionMask *graph.Node) *graph.Node) error {
	exec, err := context.NewExecAny(model.backend, model.ctx, graphFn)
	if err != nil {
		return errors.WithMessage(err, "failed to create encoder executor")
	}

	model.EncoderModel = &SubModel{
		Name:    "encoder",
		GraphFn: graphFn,
		Exec:    exec,
	}

	return nil
}

// CreateDecoderInitExec creates an executor for the decoder-init graph function.
func CreateDecoderInitExec(model *Model, graphFn any) error {
	exec, err := context.NewExecAny(model.backend, model.ctx, graphFn)
	if err != nil {
		return errors.WithMessage(err, "failed to create decoder-init executor")
	}

	model.DecoderInitModel = &SubModel{
		Name:    "decoder_init",
		GraphFn: graphFn,
		Exec:    exec,
	}

	return nil
}

// CreateDecoderExec creates an executor for the decoder graph function.
func CreateDecoderExec(model *Model, graphFn any) error {
	exec, err := context.NewExecAny(model.backend, model.ctx, graphFn)
	if err != nil {
		return errors.WithMessage(err, "failed to create decoder executor")
	}

	model.DecoderModel = &SubModel{
		Name:    "decoder",
		GraphFn: graphFn,
		Exec:    exec,
	}

	return nil
}

