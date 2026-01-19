// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package callbacks

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEarlyStopping_ModeMin(t *testing.T) {
	es := NewEarlyStopping(EarlyStoppingConfig{
		MetricName: "loss",
		Patience:   3,
		MinDelta:   0.01,
		Mode:       ModeMin,
	})

	// Initial best should be +Inf
	assert.Equal(t, math.Inf(1), es.BestValue())
	assert.False(t, es.Stopped())

	// Test improvement detection
	assert.True(t, es.isImprovement(1.0))  // Better than +Inf
	es.bestValue = 1.0

	assert.True(t, es.isImprovement(0.98))  // 1.0 - 0.98 = 0.02 > 0.01 (minDelta)
	assert.False(t, es.isImprovement(0.995)) // 1.0 - 0.995 = 0.005 < 0.01 (minDelta)
	assert.False(t, es.isImprovement(1.1))   // Getting worse
}

func TestEarlyStopping_ModeMax(t *testing.T) {
	es := NewEarlyStopping(EarlyStoppingConfig{
		MetricName: "accuracy",
		Patience:   3,
		MinDelta:   0.01,
		Mode:       ModeMax,
	})

	// Initial best should be -Inf
	assert.Equal(t, math.Inf(-1), es.BestValue())

	// Test improvement detection
	assert.True(t, es.isImprovement(0.8))  // Better than -Inf
	es.bestValue = 0.8

	assert.True(t, es.isImprovement(0.82))  // 0.82 - 0.8 = 0.02 > 0.01 (minDelta)
	assert.False(t, es.isImprovement(0.805)) // 0.805 - 0.8 = 0.005 < 0.01 (minDelta)
	assert.False(t, es.isImprovement(0.7))   // Getting worse
}

func TestEarlyStopping_Patience(t *testing.T) {
	es := NewEarlyStopping(EarlyStoppingConfig{
		MetricName: "loss",
		Patience:   3,
		Mode:       ModeMin,
	})

	// Simulate no improvement
	es.bestValue = 1.0
	es.metricIndexSet = true // Pretend we found the metric

	// Wait count should increment on no improvement
	assert.Equal(t, 0, es.waitCount)

	// Simulate no improvement multiple times
	for i := 0; i < 2; i++ {
		improved := es.isImprovement(1.1) // Not improving
		if improved {
			es.waitCount = 0
		} else {
			es.waitCount++
		}
		assert.False(t, es.stopped)
	}
	assert.Equal(t, 2, es.waitCount)

	// One more should not trigger (patience=3)
	improved := es.isImprovement(1.2)
	if !improved {
		es.waitCount++
	}
	assert.Equal(t, 3, es.waitCount)

	// Now we can mark as stopped
	if es.waitCount >= es.config.Patience {
		es.stopped = true
	}
	assert.True(t, es.Stopped())
}

func TestEarlyStopping_Reset(t *testing.T) {
	es := NewEarlyStopping(EarlyStoppingConfig{
		MetricName: "loss",
		Patience:   3,
		Mode:       ModeMin,
	})

	es.bestValue = 0.5
	es.waitCount = 2
	es.stopped = true

	es.Reset()

	assert.Equal(t, math.Inf(1), es.BestValue())
	assert.Equal(t, 0, es.waitCount)
	assert.False(t, es.Stopped())
}

func TestEarlyStopping_Baseline(t *testing.T) {
	es := NewEarlyStopping(EarlyStoppingConfig{
		MetricName:  "loss",
		Patience:    3,
		Mode:        ModeMin,
		Baseline:    0.5,
		UseBaseline: true,
	})

	// Best value should start at baseline
	assert.Equal(t, 0.5, es.BestValue())

	// Improvement is relative to baseline
	assert.True(t, es.isImprovement(0.4))  // Better than baseline
	assert.False(t, es.isImprovement(0.6)) // Worse than baseline
}

func TestBestModelCheckpoint_ModeMin(t *testing.T) {
	bm := NewBestModelCheckpoint(nil, BestModelCheckpointConfig{
		MetricName: "loss",
		Mode:       ModeMin,
		MinDelta:   0.01,
	})

	// Initial best should be +Inf
	assert.Equal(t, math.Inf(1), bm.BestValue())

	// Test improvement detection
	assert.True(t, bm.isImprovement(1.0))  // Better than +Inf
	bm.bestValue = 1.0

	assert.True(t, bm.isImprovement(0.98))  // Improvement
	assert.False(t, bm.isImprovement(0.995)) // Not enough improvement
	assert.False(t, bm.isImprovement(1.1))   // Getting worse
}

func TestBestModelCheckpoint_ModeMax(t *testing.T) {
	bm := NewBestModelCheckpoint(nil, BestModelCheckpointConfig{
		MetricName: "ndcg",
		Mode:       ModeMax,
		MinDelta:   0.01,
	})

	// Initial best should be -Inf
	assert.Equal(t, math.Inf(-1), bm.BestValue())

	// Test improvement detection
	assert.True(t, bm.isImprovement(0.8))  // Better than -Inf
	bm.bestValue = 0.8

	assert.True(t, bm.isImprovement(0.82))  // Improvement
	assert.False(t, bm.isImprovement(0.805)) // Not enough improvement
	assert.False(t, bm.isImprovement(0.7))   // Getting worse
}

func TestBestModelCheckpoint_Reset(t *testing.T) {
	bm := NewBestModelCheckpoint(nil, BestModelCheckpointConfig{
		MetricName: "loss",
		Mode:       ModeMin,
	})

	bm.bestValue = 0.5
	bm.savedCount = 3

	bm.Reset()

	assert.Equal(t, math.Inf(1), bm.BestValue())
	assert.Equal(t, 0, bm.SavedCount())
}

func TestDefaultConfigs(t *testing.T) {
	// Test EarlyStopping defaults
	es := NewEarlyStopping(EarlyStoppingConfig{
		MetricName: "loss",
	})
	assert.Equal(t, 5, es.config.Patience)   // Default patience
	assert.Equal(t, ModeMin, es.config.Mode) // Default mode

	// Test BestModelCheckpoint defaults
	bm := NewBestModelCheckpoint(nil, BestModelCheckpointConfig{
		MetricName: "loss",
	})
	assert.Equal(t, ModeMin, bm.config.Mode)
}

func TestTensorToFloat64(t *testing.T) {
	// Test nil tensor
	result := tensorToFloat64(nil)
	require.True(t, math.IsNaN(result))
}
