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
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// buildGreedyPostProcessGraph builds graph operations for greedy token selection.
// Takes logits and returns argmax token IDs.
//
// Input logits shape: [batch_size, 1, vocab_size] or [batch_size, vocab_size]
// Output shape: [batch_size] int32
func buildGreedyPostProcessGraph(logits *graph.Node) *graph.Node {
	// Handle 3D logits by squeezing the sequence dimension
	shape := logits.Shape()
	if shape.Rank() == 3 {
		// Squeeze middle dimension: [B, 1, V] -> [B, V]
		logits = graph.Squeeze(logits, 1)
	}

	// ArgMax on last axis (vocabulary dimension)
	tokenIDs := graph.ArgMax(logits, -1, dtypes.Int32)
	return tokenIDs
}

// buildSamplingPostProcessGraph builds graph operations for sampling preparation.
// Applies temperature scaling, softmax, and TopK filtering on-device.
// Returns TopK probabilities and indices for CPU-side sampling.
//
// Input logits shape: [batch_size, 1, vocab_size] or [batch_size, vocab_size]
// Input temperature: scalar float32
// Output topKProbs shape: [batch_size, k] float32
// Output topKIndices shape: [batch_size, k] int32
func buildSamplingPostProcessGraph(logits, temperature *graph.Node, k int) (topKProbs, topKIndices *graph.Node) {
	shape := logits.Shape()
	if shape.Rank() == 3 {
		// Squeeze middle dimension: [B, 1, V] -> [B, V]
		logits = graph.Squeeze(logits, 1)
	}

	// Apply temperature scaling: logits / temperature
	// Broadcast scalar temperature across all logits
	scaledLogits := graph.Div(logits, temperature)

	// Softmax to get probabilities (on last axis)
	probs := graph.Softmax(scaledLogits, -1)

	// TopK to get top k probabilities and their indices
	// Returns values in descending order
	topKProbs, topKIndices = graph.TopK(probs, k, -1)

	return topKProbs, topKIndices
}
