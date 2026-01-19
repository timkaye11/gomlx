// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package callbacks

import (
	"fmt"
	"math"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/pkg/errors"
)

// BestModelCheckpointConfig configures the best model checkpoint callback.
type BestModelCheckpointConfig struct {
	// MetricName is the name of the metric to monitor.
	MetricName string

	// Mode determines whether to minimize or maximize the metric.
	// Use ModeMin for loss, ModeMax for accuracy/NDCG.
	// Default: ModeMin
	Mode MonitorMode

	// MinDelta is the minimum change to qualify as an improvement.
	// Default: 0
	MinDelta float64

	// Verbose prints a message when a new best model is saved.
	Verbose bool
}

// BestModelCheckpoint saves the model only when a monitored metric improves.
//
// This is useful for fine-tuning where you want to keep only the best model
// checkpoint instead of the latest one.
type BestModelCheckpoint struct {
	config              BestModelCheckpointConfig
	checkpoint          *checkpoints.Handler
	checkpointInterface Checkpointer // Alternative interface for testing
	bestValue           float64
	metricIndex         int
	metricIndexSet      bool
	useTrainMetrics     bool
	savedCount          int
}

// NewBestModelCheckpoint creates a new best model checkpoint callback.
//
// Example:
//
//	bestModel := callbacks.NewBestModelCheckpoint(checkpoint, callbacks.BestModelCheckpointConfig{
//	    MetricName: "eval_ndcg",
//	    Mode:       callbacks.ModeMax,
//	})
//	train.EveryNSteps(loop, 100, "best_model", 100, bestModel.OnStepFn)
func NewBestModelCheckpoint(checkpoint *checkpoints.Handler, config BestModelCheckpointConfig) *BestModelCheckpoint {
	if config.Mode == "" {
		config.Mode = ModeMin
	}

	bestValue := math.Inf(1) // For min mode
	if config.Mode == ModeMax {
		bestValue = math.Inf(-1) // For max mode
	}

	return &BestModelCheckpoint{
		config:     config,
		checkpoint: checkpoint,
		bestValue:  bestValue,
	}
}

// OnStepFn implements train.OnStepFn for use with Loop.OnStep or EveryNSteps.
//
// It saves the model only when the monitored metric improves.
func (bm *BestModelCheckpoint) OnStepFn(loop *train.Loop, stepMetrics []*tensors.Tensor) error {
	if bm.checkpoint == nil && bm.checkpointInterface == nil {
		return nil
	}

	// Find the metric index on first call
	if !bm.metricIndexSet {
		if err := bm.findMetricIndex(loop.Trainer); err != nil {
			return err
		}
	}

	// Get the metric value
	var metricValue float64
	if bm.useTrainMetrics {
		if bm.metricIndex >= len(stepMetrics) {
			return errors.Errorf("BestModelCheckpoint: metric index %d out of range (have %d train metrics)",
				bm.metricIndex, len(stepMetrics))
		}
		metricValue = tensorToFloat64(stepMetrics[bm.metricIndex])
	} else {
		// For eval metrics, check SharedData
		if evalMetrics, ok := loop.SharedData["eval_metrics"].([]*tensors.Tensor); ok {
			if bm.metricIndex >= len(evalMetrics) {
				return errors.Errorf("BestModelCheckpoint: metric index %d out of range (have %d eval metrics)",
					bm.metricIndex, len(evalMetrics))
			}
			metricValue = tensorToFloat64(evalMetrics[bm.metricIndex])
		} else {
			// Try using train metrics if metric is there
			if bm.metricIndex < len(stepMetrics) {
				metricValue = tensorToFloat64(stepMetrics[bm.metricIndex])
			} else {
				return errors.Errorf("BestModelCheckpoint: cannot find eval_metrics in SharedData and train metrics don't include %q",
					bm.config.MetricName)
			}
		}
	}

	// Check for improvement
	if bm.isImprovement(metricValue) {
		bm.bestValue = metricValue
		bm.savedCount++
		if bm.config.Verbose {
			fmt.Printf("BestModelCheckpoint: new best %s=%f at step %d (save #%d)\n",
				bm.config.MetricName, metricValue, loop.LoopStep, bm.savedCount)
		}
		return bm.save()
	}

	return nil
}

// save saves using either the Handler or the interface.
func (bm *BestModelCheckpoint) save() error {
	if bm.checkpointInterface != nil {
		return bm.checkpointInterface.Save()
	}
	if bm.checkpoint != nil {
		return bm.checkpoint.Save()
	}
	return nil
}

// findMetricIndex finds the index of the monitored metric in the trainer's metrics.
func (bm *BestModelCheckpoint) findMetricIndex(trainer *train.Trainer) error {
	// First try eval metrics
	for i, m := range trainer.EvalMetrics() {
		if m.Name() == bm.config.MetricName {
			bm.metricIndex = i
			bm.metricIndexSet = true
			bm.useTrainMetrics = false
			return nil
		}
	}

	// Then try train metrics
	for i, m := range trainer.TrainMetrics() {
		if m.Name() == bm.config.MetricName {
			bm.metricIndex = i
			bm.metricIndexSet = true
			bm.useTrainMetrics = true
			return nil
		}
	}

	return errors.Errorf("BestModelCheckpoint: metric %q not found in train or eval metrics", bm.config.MetricName)
}

// isImprovement checks if the new value is an improvement over the best value.
func (bm *BestModelCheckpoint) isImprovement(newValue float64) bool {
	if bm.config.Mode == ModeMin {
		return newValue < bm.bestValue-bm.config.MinDelta
	}
	return newValue > bm.bestValue+bm.config.MinDelta
}

// BestValue returns the best observed metric value.
func (bm *BestModelCheckpoint) BestValue() float64 {
	return bm.bestValue
}

// SavedCount returns the number of times the model was saved.
func (bm *BestModelCheckpoint) SavedCount() int {
	return bm.savedCount
}

// Reset resets the callback state.
func (bm *BestModelCheckpoint) Reset() {
	bm.savedCount = 0
	if bm.config.Mode == ModeMin {
		bm.bestValue = math.Inf(1)
	} else {
		bm.bestValue = math.Inf(-1)
	}
}
