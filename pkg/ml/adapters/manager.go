// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"sync"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// ParamAdapterManager is the context parameter key for the adapter manager.
// Layers can check for this parameter to apply adapters.
const ParamAdapterManager = "adapter_manager"

// Manager handles multiple adapters per model with activation/deactivation support.
// It allows hot-swapping between different adapters without reloading the base model.
type Manager struct {
	mu       sync.RWMutex
	adapters map[string]Adapter // name -> adapter
	active   string             // currently active adapter name (empty = none)
	ctx      *context.Context   // root context for adapter storage
}

// NewManager creates a new adapter manager attached to the given context.
// The manager stores adapter parameters under ctx.In("adapters").
func NewManager(ctx *context.Context) *Manager {
	return &Manager{
		adapters: make(map[string]Adapter),
		ctx:      ctx.In("adapters"),
	}
}

// Add registers a new adapter with the manager.
// Returns an error if an adapter with the same name already exists.
func (m *Manager) Add(adapter Adapter) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	name := adapter.Name()
	if _, exists := m.adapters[name]; exists {
		return errors.Errorf("adapter %q already exists", name)
	}

	m.adapters[name] = adapter
	return nil
}

// Remove removes an adapter by name.
// Returns an error if the adapter doesn't exist or is currently active.
func (m *Manager) Remove(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.adapters[name]; !exists {
		return errors.Errorf("adapter %q not found", name)
	}

	if m.active == name {
		return errors.Errorf("cannot remove active adapter %q; deactivate it first", name)
	}

	delete(m.adapters, name)
	return nil
}

// Get retrieves an adapter by name.
// Returns nil and false if the adapter doesn't exist.
func (m *Manager) Get(name string) (Adapter, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	adapter, exists := m.adapters[name]
	return adapter, exists
}

// SetActive activates the named adapter and deactivates others.
// Pass empty string to deactivate all adapters.
// Returns an error if the adapter doesn't exist.
func (m *Manager) SetActive(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if name != "" {
		if _, exists := m.adapters[name]; !exists {
			return errors.Errorf("adapter %q not found", name)
		}
	}

	m.active = name
	return nil
}

// Active returns the currently active adapter, or nil if none.
func (m *Manager) Active() Adapter {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.active == "" {
		return nil
	}
	return m.adapters[m.active]
}

// ActiveName returns the name of the currently active adapter, or empty string if none.
func (m *Manager) ActiveName() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.active
}

// List returns names of all registered adapters.
func (m *Manager) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.adapters))
	for name := range m.adapters {
		names = append(names, name)
	}
	return names
}

// Count returns the number of registered adapters.
func (m *Manager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.adapters)
}

// Apply applies the active adapter (if any) to the given computation.
// This is the main entry point for layer integration.
//
// If no adapter is active or the active adapter is disabled, returns baseOutput unchanged.
//
// Parameters:
//   - ctx: Layer-scoped context for variable creation
//   - input: Input tensor [batch, ..., inputDim]
//   - baseOutput: Output from base layer [batch, ..., outputDim]
//   - inputDim, outputDim: Dimensions for adapter parameter creation
//
// Returns: Modified output with adapter applied
func (m *Manager) Apply(ctx *context.Context, input, baseOutput *Node, inputDim, outputDim int) *Node {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.active == "" {
		return baseOutput
	}

	adapter := m.adapters[m.active]
	if adapter == nil || !adapter.Config().Enabled() {
		return baseOutput
	}

	// Scope the context under the adapter name for variable namespacing
	adapterCtx := ctx.In(m.active)
	return adapter.Apply(adapterCtx, input, baseOutput, inputDim, outputDim)
}

// Context returns the context used for adapter variable storage.
func (m *Manager) Context() *context.Context {
	return m.ctx
}

// FreezeBaseWeights freezes base weights for the active adapter.
func (m *Manager) FreezeBaseWeights() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.active != "" {
		if adapter := m.adapters[m.active]; adapter != nil {
			adapter.FreezeBaseWeights(m.ctx)
		}
	}
}

// CountActiveParameters returns the number of trainable parameters in the active adapter.
// Returns 0 if no adapter is active.
func (m *Manager) CountActiveParameters() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.active == "" {
		return 0
	}

	adapter := m.adapters[m.active]
	if adapter == nil {
		return 0
	}

	return adapter.CountParameters(m.ctx)
}

// GetFromContext retrieves the adapter manager from a context's parameters.
// Returns nil if no manager is configured.
func GetFromContext(ctx *context.Context) *Manager {
	return context.GetParamOr(ctx, ParamAdapterManager, (*Manager)(nil))
}

// SetInContext stores the adapter manager in a context's parameters.
func SetInContext(ctx *context.Context, manager *Manager) {
	ctx.SetParam(ParamAdapterManager, manager)
}
