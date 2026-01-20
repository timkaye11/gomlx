// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dora

import (
	"github.com/gomlx/gomlx/pkg/ml/adapters"
	"github.com/pkg/errors"
)

// Factory implements adapters.AdapterFactory for DoRA.
type Factory struct{}

// NewFactory creates a new DoRA factory.
func NewFactory() *Factory {
	return &Factory{}
}

// Type returns the adapter type this factory creates.
func (f *Factory) Type() adapters.AdapterType {
	return adapters.AdapterTypeDoRA
}

// Create creates a new DoRA adapter with the given name and configuration.
func (f *Factory) Create(name string, config adapters.AdapterConfig) (adapters.Adapter, error) {
	doraConfig, ok := config.(*Config)
	if !ok {
		return nil, errors.Errorf("expected *dora.Config, got %T", config)
	}
	return New(name, doraConfig)
}

// DefaultConfig returns a default DoRA configuration.
func (f *Factory) DefaultConfig() adapters.AdapterConfig {
	return NewConfig()
}

func init() {
	// Register DoRA factory with the global registry
	_ = adapters.Register(NewFactory())
}
