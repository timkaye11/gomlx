// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ia3

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/pkg/errors"
)

// Factory creates IA³ adapter instances.
type Factory struct{}

// NewFactory creates a new IA³ factory.
func NewFactory() *Factory {
	return &Factory{}
}

// Create returns a new IA³ adapter instance.
func (f *Factory) Create(name string, config adapters.AdapterConfig) (adapters.Adapter, error) {
	if config == nil {
		return New(name, nil)
	}

	ia3Config, ok := config.(*Config)
	if !ok {
		return nil, errors.Errorf("IA³ factory requires *ia3.Config, got %T", config)
	}

	return New(name, ia3Config)
}

// Type returns the adapter type this factory creates.
func (f *Factory) Type() adapters.AdapterType {
	return adapters.AdapterTypeIA3
}

func init() {
	// Register the IA³ factory with the global registry
	adapters.MustRegister(NewFactory())
}
