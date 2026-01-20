// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"sync"

	"github.com/pkg/errors"
)

// globalRegistry is the default factory registry.
var globalRegistry = NewRegistry()

// Registry manages adapter factories by type.
// It provides a factory pattern for creating adapters from configurations.
type Registry struct {
	mu        sync.RWMutex
	factories map[AdapterType]AdapterFactory
}

// NewRegistry creates a new empty adapter registry.
func NewRegistry() *Registry {
	return &Registry{
		factories: make(map[AdapterType]AdapterFactory),
	}
}

// Register adds a factory to the registry.
// Returns an error if a factory for the same type already exists.
func (r *Registry) Register(factory AdapterFactory) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	adapterType := factory.Type()
	if _, exists := r.factories[adapterType]; exists {
		return errors.Errorf("adapter factory for type %q already registered", adapterType)
	}

	r.factories[adapterType] = factory
	return nil
}

// MustRegister adds a factory to the registry, panicking on error.
// This is useful for init() functions.
func (r *Registry) MustRegister(factory AdapterFactory) {
	if err := r.Register(factory); err != nil {
		panic(err)
	}
}

// Unregister removes a factory from the registry.
func (r *Registry) Unregister(adapterType AdapterType) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.factories, adapterType)
}

// Get retrieves a factory by adapter type.
// Returns nil and false if not found.
func (r *Registry) Get(adapterType AdapterType) (AdapterFactory, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	factory, exists := r.factories[adapterType]
	return factory, exists
}

// Create creates an adapter using the appropriate factory.
// Returns an error if no factory is registered for the config's type.
func (r *Registry) Create(name string, config AdapterConfig) (Adapter, error) {
	r.mu.RLock()
	factory, exists := r.factories[config.Type()]
	r.mu.RUnlock()

	if !exists {
		return nil, errors.Errorf("no factory registered for adapter type %q", config.Type())
	}

	return factory.Create(name, config)
}

// Types returns all registered adapter types.
func (r *Registry) Types() []AdapterType {
	r.mu.RLock()
	defer r.mu.RUnlock()

	types := make([]AdapterType, 0, len(r.factories))
	for t := range r.factories {
		types = append(types, t)
	}
	return types
}

// Count returns the number of registered factories.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.factories)
}

// Global registry functions

// Register adds a factory to the global registry.
func Register(factory AdapterFactory) error {
	return globalRegistry.Register(factory)
}

// MustRegister adds a factory to the global registry, panicking on error.
func MustRegister(factory AdapterFactory) {
	globalRegistry.MustRegister(factory)
}

// Unregister removes a factory from the global registry.
func Unregister(adapterType AdapterType) {
	globalRegistry.Unregister(adapterType)
}

// GetFactory retrieves a factory from the global registry.
func GetFactory(adapterType AdapterType) (AdapterFactory, bool) {
	return globalRegistry.Get(adapterType)
}

// CreateAdapter creates an adapter using the global registry.
func CreateAdapter(name string, config AdapterConfig) (Adapter, error) {
	return globalRegistry.Create(name, config)
}

// RegisteredTypes returns all types registered in the global registry.
func RegisteredTypes() []AdapterType {
	return globalRegistry.Types()
}

// GlobalRegistry returns the global registry instance.
// Use this when you need direct access to the registry.
func GlobalRegistry() *Registry {
	return globalRegistry
}
