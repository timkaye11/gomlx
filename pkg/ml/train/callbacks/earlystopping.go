// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package callbacks provides training callbacks for common fine-tuning patterns.
//
// Callbacks include:
//   - EarlyStopping: Stop training when a monitored metric stops improving
//   - BestModelCheckpoint: Save model only when a monitored metric improves
package callbacks

import (
	"fmt"
	"math"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/pkg/errors"
)

// ErrEarlyStopped is returned when training is stopped early due to no improvement.
var ErrEarlyStopped = errors.New("early stopping triggered")

// MonitorMode specifies whether to minimize or maximize the monitored metric.
type MonitorMode string

const (
	// ModeMin monitors for decreasing metric values (e.g., loss).
	ModeMin MonitorMode = "min"
	// ModeMax monitors for increasing metric values (e.g., accuracy, NDCG).
	ModeMax MonitorMode = "max"
)

// EarlyStoppingConfig configures the early stopping callback.
type EarlyStoppingConfig struct {
	// MetricName is the name of the metric to monitor.
	// This should match the Name() of one of the metrics in the Trainer.
	MetricName string

	// Patience is the number of evaluations with no improvement after which training stops.
	// Default: 5
	Patience int

	// MinDelta is the minimum change to qualify as an improvement.
	// For mode="min", improvement means value decreased by at least MinDelta.
	// For mode="max", improvement means value increased by at least MinDelta.
	// Default: 0
	MinDelta float64

	// Mode determines whether to minimize or maximize the metric.
	// Use ModeMin for loss, ModeMax for accuracy/NDCG.
	// Default: ModeMin
	Mode MonitorMode

	// Baseline is an optional initial value to compare against.
	// If set, training stops if the metric doesn't improve beyond this baseline.
	// If not set (0), uses the first observed value as baseline.
	Baseline float64

	// UseBaseline indicates whether to use the Baseline value.
	UseBaseline bool

	// Verbose prints a message when early stopping is triggered.
	Verbose bool
}

// EarlyStopping implements the early stopping callback.
type EarlyStopping struct {
	config          EarlyStoppingConfig
	bestValue       float64
	waitCount       int
	stopped         bool
	metricIndex     int  // Index of the metric in the trainer's metric list
	metricIndexSet  bool // Whether we've found the metric index
	useTrainMetrics bool // Whether to use train metrics (true) or eval metrics (false)
}

// NewEarlyStopping creates a new early stopping callback.
//
// Example:
//
//	earlyStop := callbacks.NewEarlyStopping(callbacks.EarlyStoppingConfig{
//	    MetricName: "eval_loss",
//	    Patience:   5,
//	    MinDelta:   0.001,
//	    Mode:       callbacks.ModeMin,
//	})
//	train.EveryNSteps(loop, 100, "early_stop", 100, earlyStop.OnStepFn)
func NewEarlyStopping(config EarlyStoppingConfig) *EarlyStopping {
	// Set defaults
	if config.Patience <= 0 {
		config.Patience = 5
	}
	if config.Mode == "" {
		config.Mode = ModeMin
	}

	// Initialize best value based on mode
	bestValue := math.Inf(1) // For min mode
	if config.Mode == ModeMax {
		bestValue = math.Inf(-1) // For max mode
	}
	if config.UseBaseline {
		bestValue = config.Baseline
	}

	return &EarlyStopping{
		config:    config,
		bestValue: bestValue,
	}
}

// OnStepFn implements train.OnStepFn for use with Loop.OnStep or EveryNSteps.
//
// Note: This callback expects to receive metrics from an evaluation step.
// It should be used after an evaluation callback that populates the metrics.
// Alternatively, use OnEvalFn with a custom evaluation callback.
func (es *EarlyStopping) OnStepFn(loop *train.Loop, stepMetrics []*tensors.Tensor) error {
	if es.stopped {
		return ErrEarlyStopped
	}

	// Find the metric index on first call
	if !es.metricIndexSet {
		if err := es.findMetricIndex(loop.Trainer); err != nil {
			return err
		}
	}

	// Get the metric value
	var metricValue float64
	if es.useTrainMetrics {
		if es.metricIndex >= len(stepMetrics) {
			return errors.Errorf("EarlyStopping: metric index %d out of range (have %d train metrics)",
				es.metricIndex, len(stepMetrics))
		}
		metricValue = tensorToFloat64(stepMetrics[es.metricIndex])
	} else {
		// For eval metrics, we need to get them from SharedData or run evaluation
		// Check if there's eval data in SharedData
		if evalMetrics, ok := loop.SharedData["eval_metrics"].([]*tensors.Tensor); ok {
			if es.metricIndex >= len(evalMetrics) {
				return errors.Errorf("EarlyStopping: metric index %d out of range (have %d eval metrics)",
					es.metricIndex, len(evalMetrics))
			}
			metricValue = tensorToFloat64(evalMetrics[es.metricIndex])
		} else {
			// Try using train metrics if metric is there
			if es.metricIndex < len(stepMetrics) {
				metricValue = tensorToFloat64(stepMetrics[es.metricIndex])
			} else {
				return errors.Errorf("EarlyStopping: cannot find eval_metrics in SharedData and train metrics don't include %q",
					es.config.MetricName)
			}
		}
	}

	// Check for improvement
	improved := es.isImprovement(metricValue)
	if improved {
		es.bestValue = metricValue
		es.waitCount = 0
	} else {
		es.waitCount++
		if es.waitCount >= es.config.Patience {
			es.stopped = true
			if es.config.Verbose {
				fmt.Printf("EarlyStopping: stopped at step %d after %d evaluations without improvement (best=%f, current=%f)\n",
					loop.LoopStep, es.config.Patience, es.bestValue, metricValue)
			}
			return ErrEarlyStopped
		}
	}

	return nil
}

// findMetricIndex finds the index of the monitored metric in the trainer's metrics.
func (es *EarlyStopping) findMetricIndex(trainer *train.Trainer) error {
	// First try eval metrics
	for i, m := range trainer.EvalMetrics() {
		if m.Name() == es.config.MetricName {
			es.metricIndex = i
			es.metricIndexSet = true
			es.useTrainMetrics = false
			return nil
		}
	}

	// Then try train metrics
	for i, m := range trainer.TrainMetrics() {
		if m.Name() == es.config.MetricName {
			es.metricIndex = i
			es.metricIndexSet = true
			es.useTrainMetrics = true
			return nil
		}
	}

	return errors.Errorf("EarlyStopping: metric %q not found in train or eval metrics", es.config.MetricName)
}

// isImprovement checks if the new value is an improvement over the best value.
func (es *EarlyStopping) isImprovement(newValue float64) bool {
	if es.config.Mode == ModeMin {
		return newValue < es.bestValue-es.config.MinDelta
	}
	return newValue > es.bestValue+es.config.MinDelta
}

// Stopped returns true if early stopping has been triggered.
func (es *EarlyStopping) Stopped() bool {
	return es.stopped
}

// BestValue returns the best observed metric value.
func (es *EarlyStopping) BestValue() float64 {
	return es.bestValue
}

// Reset resets the early stopping state.
func (es *EarlyStopping) Reset() {
	es.stopped = false
	es.waitCount = 0
	if es.config.Mode == ModeMin {
		es.bestValue = math.Inf(1)
	} else {
		es.bestValue = math.Inf(-1)
	}
	if es.config.UseBaseline {
		es.bestValue = es.config.Baseline
	}
}

// GetMetricByName finds a metric by name from the trainer.
// Returns the metric interface and its index, or an error if not found.
func GetMetricByName(trainer *train.Trainer, name string) (metrics.Interface, int, bool, error) {
	// Try eval metrics first
	for i, m := range trainer.EvalMetrics() {
		if m.Name() == name {
			return m, i, false, nil // false = not train metric
		}
	}
	// Then try train metrics
	for i, m := range trainer.TrainMetrics() {
		if m.Name() == name {
			return m, i, true, nil // true = train metric
		}
	}
	return nil, -1, false, errors.Errorf("metric %q not found", name)
}

// tensorToFloat64 extracts a scalar float64 value from a tensor.
func tensorToFloat64(t *tensors.Tensor) float64 {
	if t == nil {
		return math.NaN()
	}
	return tensors.ToScalar[float64](t)
}
