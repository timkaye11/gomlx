---
Refined Implementation Plan: ORT-Style Dynamic Shapes for GoMLX

Executive Summary

Based on thorough codebase analysis, the original plan needs significant refinement. The key insight is that GoMLX's architecture fundamentally freezes shapes at graph build time, but several pragmatic approaches exist that minimize disruption while enabling flexible batch/sequence dimensions.

---
Critical Findings

1. Shape System Reality

- shapes.Shape uses Dimensions []int - all static integers
- -1 wildcard already exists in assertions (AssertDims) but NOT in actual shapes
- Shape comparison uses exact equality via slices.Equal()
- Recommendation: Use type alias type Dimension int with negative values for dynamic dims (backward compatible)

2. Graph Caching Architecture

- O(n) linear scan with exact shape matching (no hashing)
- Default cache size: 32 entries (DefaultExecMaxCacheSize)
- Different shapes = different compiled graphs (this is fundamental)
- Pattern matching IS compatible as pre-filtering layer

3. SimpleGo Backend Reality

- Shapes frozen at Builder.newNode() call (line 99)
- No true shape specialization - same code path for all shapes
- "Specialization" means graph rebuilding per shape, not single-graph flexibility
- The existing Exec caching already handles multiple shapes via graph-per-shape

4. Existing Dynamic Patterns

- Ragged2D: Elegant solution for variable sequences (first dim static)
- DynamicSlice/DynamicUpdateSlice: Dynamic start positions, static output sizes
- Masking: Universal pattern for variable-length inputs
- -1 in assertions: Already supports "unchecked" dimension concept

5. Gradient Computation Challenges

- VJP functions assume static shapes for reconstruction
- ReduceSum, BroadcastInDim, Gather need shape capture
- Shape assertions in rev_autodiff.go will fail with unknowns
- Solution: Store forward-pass shapes in node metadata

---
Revised Three-Phase Approach

Phase 1: Foundation - Symbolic Dimension Type

Goal: Add symbolic dimension representation without breaking existing code

Task 1.1: Dimension Type Alias

File: pkg/core/shapes/shapes.go

// Dimension represents either a static or symbolic dimension
type Dimension int  // Negative values = symbolic, Positive = static

const (
    DimBatch     Dimension = -1  // "batch"
    DimSeqLen    Dimension = -2  // "seq_len"
    DimUnknown   Dimension = -3  // generic unknown
    // Reserve -1 to -100 for named dynamics
)

func (d Dimension) IsStatic() bool   { return d > 0 }
func (d Dimension) Value() int       { return int(d) }
func (d Dimension) Name() string     { /* map negative to name */ }
func (d Dimension) String() string   { /* human-readable */ }

Why type alias:
- shape.Dimensions[0] still returns Dimension which is int
- slices.Equal() still works
- Gob serialization works natively
- 100% backward compatible

Task 1.2: Enhanced Shape Equality

File: pkg/core/shapes/shapes.go

// Equal returns true if shapes match exactly (static) or symbolically (dynamic)
func (s Shape) Equal(s2 Shape) bool {
    // ... existing dtype/rank checks ...

    for i, d := range s.Dimensions {
        d2 := s2.Dimensions[i]
        switch {
        case d > 0 && d2 > 0:
            if d != d2 { return false }  // Both static, must match
        case d < 0 && d2 < 0:
            if d != d2 { return false }  // Both symbolic, must be same symbol
        default:
            return false  // One static, one dynamic = not equal
        }
    }
    return true
}

// Matches checks if concrete shape matches a pattern with symbolic dims
func (s Shape) Matches(pattern Shape) bool {
    // ... dtype/rank checks ...
    for i, d := range s.Dimensions {
        pd := pattern.Dimensions[i]
        if pd < 0 { continue }  // Symbolic matches anything
        if d != pd { return false }
    }
    return true
}

Task 1.3: Shape Constructors

File: pkg/core/shapes/shapes.go

// MakeDynamic creates shape with symbolic dimensions
func MakeDynamic(dtype dtypes.DType, dims ...Dimension) Shape {
    // Validate: symbolic dims allowed, negative non-symbolic not allowed
    return Shape{DType: dtype, Dimensions: dims}
}

// WithDynamicBatch replaces first dimension with DimBatch
func (s Shape) WithDynamicBatch() Shape {
    clone := s.Clone()
    if len(clone.Dimensions) > 0 {
        clone.Dimensions[0] = int(DimBatch)
    }
    return clone
}

Task 1.4: Update Assertions

File: pkg/core/shapes/asserts.go

// Existing -1 wildcard already works, but clarify semantics
// -1 in AssertDims = "any value" (existing)
// DimBatch (-1) in Shape = "dynamic batch dimension" (new)
// These are different uses of -1 that happen to align

---
Phase 2: Shape Inference with Symbolic Dimensions

Goal: Propagate symbolic dimensions through operations

Task 2.1: Basic Operation Inference

File: backends/shapeinference/shapeinference.go

Operations that require NO changes (shape-preserving):
- All StandardUnaryOperations (Abs, Sqrt, Sin, etc.)

Operations with trivial changes (dimension manipulation):
func BinaryOp(opType OpType, lhs, rhs Shape) (Shape, error) {
    output := Shape{DType: resultDType}
    for axis := range output.Rank() {
        ld, rd := lhs.Dimensions[axis], rhs.Dimensions[axis]
        // Handle symbolic dimensions
        switch {
        case ld < 0 && rd < 0 && ld == rd:
            output.Dimensions[axis] = ld  // Same symbol
        case ld < 0 && rd == 1:
            output.Dimensions[axis] = ld  // Symbolic wins over broadcast
        case rd < 0 && ld == 1:
            output.Dimensions[axis] = rd  // Symbolic wins over broadcast
        case ld > 0 && rd > 0:
            output.Dimensions[axis] = max(ld, rd)  // Existing logic
        default:
            return Shape{}, errors.New("incompatible symbolic dimensions")
        }
    }
    return output, nil
}

Task 2.2: Operation Difficulty Classification

Based on exploration, prioritize by difficulty:

TRIVIAL (implement immediately):
- Unary ops (47 total)
- BinaryOp/ComparisonOp
- Transpose, Reshape (with size validation)
- ReduceOp variants
- Concatenate

EASY (arithmetic on static params):
- ConvGeneralOp (strides/dilations are static)
- ReduceWindow
- DotGeneral
- Pad

MODERATE (complex validation):
- Gather (sliceSizes static, batch dims symbolic)
- ScatterOp
- BroadcastInDim

HARD (require deferred validation):
- Slice (start/limit must be static)
- FFT

Task 2.3: Constraint Propagation

New file: pkg/core/graph/constraints.go

type ConstraintContext struct {
    equalities map[string]string  // dimA == dimB
    resolved   map[string]int     // dimName -> known value
}

// AddEquality records that two symbolic dims must be equal
func (c *ConstraintContext) AddEquality(a, b Dimension) error

// Resolve binds a symbolic dimension to a concrete value
func (c *ConstraintContext) Resolve(dim Dimension, value int) error

// Validate checks all constraints are satisfiable
func (c *ConstraintContext) Validate() error

This allows detecting errors like batch being used inconsistently.

---
Phase 3: Graph Caching with Shape Patterns

Goal: Reduce graph compilations via pattern-based caching

Task 3.1: Pattern-Based Cache Lookup

File: pkg/core/graph/exec.go

type ShapePattern struct {
    DType       dtypes.DType
    Dimensions  []int  // Negative = symbolic, positive = exact
}

type execGraphCacheEntry struct {
    exactShapes   []shapes.Shape   // For exact match (fast path)
    shapePattern  []ShapePattern   // For pattern match (fallback)
    graph         *Graph
    // ... existing fields
}

func (e *Exec) findCachedGraph(argsShapes []shapes.Shape) *execGraphCacheEntry {
    // Fast path: exact match (existing logic)
    for _, entry := range e.cache {
        if exactMatch(entry.exactShapes, argsShapes) {
            return entry
        }
    }

    // Slow path: pattern match (NEW)
    for _, entry := range e.patternCache {
        if patternMatch(entry.shapePattern, argsShapes) {
            return entry
        }
    }

    return nil  // Must build new graph
}

Task 3.2: Automatic Bucketing

File: pkg/core/graph/exec.go

type ExecOptions struct {
    // Existing options...

    // Dynamic shape support
    EnablePatternCaching bool
    BucketingStrategy    BucketingStrategy  // Pow2, Linear, Custom
    MaxCachePerPattern   int
}

type BucketingStrategy interface {
    Bucket(dim int) int  // Returns bucketed dimension
}

// Power-of-2 bucketing (most common)
type Pow2Bucketing struct{}
func (Pow2Bucketing) Bucket(dim int) int {
    if dim <= 0 { return dim }  // Preserve symbolic
    return 1 << bits.Len(uint(dim-1))  // Round up to power of 2
}

Task 3.3: Runtime Shape Binding

File: pkg/core/graph/exec.go

When calling Exec with actual shapes:
1. Check exact cache → hit? execute
2. Check pattern cache → hit? validate bindings, execute
3. No hit? Build graph for bucketed shape, cache with pattern

func (e *Exec) Call(args ...*tensors.Tensor) []*tensors.Tensor {
    argsShapes := extractShapes(args)

    // Try exact match first
    if entry := e.findExactMatch(argsShapes); entry != nil {
        return e.execute(entry, args)
    }

    // Try pattern match with bucketing
    if e.opts.EnablePatternCaching {
        bucketedShapes := e.applyBucketing(argsShapes)
        if entry := e.findPatternMatch(bucketedShapes); entry != nil {
            // May need to pad inputs to bucketed size
            paddedArgs := e.padToShape(args, bucketedShapes)
            return e.execute(entry, paddedArgs)
        }
    }

    // Build new graph
    return e.buildAndExecute(argsShapes)
}

---
Phase 4: Gradient Computation Support

Goal: Make backprop work with symbolic dimensions

Task 4.1: Shape Capture in VJP

File: pkg/core/graph/rev_autodiff.go

For operations that reconstruct shapes in gradients:

func reduceSumVJP(node, v *Node, outputShape shapes.Shape) []*Node {
    params := node.inputs.(*nodeInputsReduceSum)
    x := params.x

    // CHANGE: Use captured input shape, not current shape
    inputShape := node.GetCapturedInputShape(0)  // NEW method

    // Reconstruct reduced dimensions
    newShape := inputShape.Clone()
    for _, dim := range params.axes {
        newShape.Dimensions[dim] = 1
    }
    // ... rest unchanged
}

Task 4.2: Forward Shape Capture

File: pkg/core/graph/node.go

type Node struct {
    // ... existing fields ...

    // For gradient computation with dynamic shapes
    capturedInputShapes []shapes.Shape  // Snapshot of input shapes at build time
}

// During graph building, capture shapes
func (g *Graph) newNode(opType OpType, shape Shape, inputs ...*Node) *Node {
    n := &Node{/* ... */}

    // Capture input shapes for gradient computation
    n.capturedInputShapes = make([]shapes.Shape, len(inputs))
    for i, input := range inputs {
        n.capturedInputShapes[i] = input.Shape()
    }

    return n
}

Task 4.3: Relaxed Shape Assertions

File: pkg/core/graph/rev_autodiff.go

// Line 215-224: Modified assertion
combinedShape := combineOutputShape(outputShape, input.Shape())

// CHANGE: Allow symbolic dimension matching
if !vjp.Shape().MatchesPattern(combinedShape) {
    // Only fail if concrete dimensions mismatch
    Panicf("invalid gradient shape: %s vs %s", vjp.Shape(), combinedShape)
}

---
Operation Support Matrix

| Operation    | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Notes                    |
|--------------|---------|---------|---------|---------|--------------------------|
| Unary ops    | ✅      | ✅      | ✅      | ✅      | No changes needed        |
| Binary ops   | ✅      | ✅      | ✅      | ✅      | Symbolic max()           |
| Transpose    | ✅      | ✅      | ✅      | ✅      | Index manipulation       |
| Reshape      | ⚠️      | ✅      | ✅      | ✅      | Total size validation    |
| Reduce*      | ✅      | ✅      | ✅      | ⚠️      | VJP needs shape capture  |
| Concat       | ✅      | ✅      | ✅      | ✅      | Symbolic addition        |
| Conv         | ✅      | ✅      | ✅      | ✅      | Static params            |
| DotGeneral   | ✅      | ✅      | ✅      | ✅      | Axis selection           |
| Gather       | ⚠️      | ⚠️      | ⚠️      | ⚠️      | Complex validation       |
| Broadcast    | ✅      | ✅      | ✅      | ⚠️      | VJP needs axis metadata  |
| DynamicSlice | ✅      | ✅      | ✅      | ✅      | Already supports dynamic |

---
Files to Modify (by priority)

High Priority

1. pkg/core/shapes/shapes.go - Dimension type, equality methods
2. backends/shapeinference/shapeinference.go - Symbolic inference
3. pkg/core/graph/exec.go - Pattern caching, bucketing
4. pkg/core/graph/node.go - Shape capture for gradients

Medium Priority

5. pkg/core/shapes/asserts.go - Enhanced assertions
6. pkg/core/graph/rev_autodiff.go - VJP shape handling
7. pkg/core/graph/constraints.go - NEW: constraint propagation
8. backends/simplego/builder.go - Parameter shape handling

Lower Priority

9. pkg/ml/context/variables.go - Variable shape tracking
10. pkg/ml/context/checkpoints/ - Checkpoint format v2

---
Key Differences from Original Plan

1. Type alias not struct: Use type Dimension int instead of struct{Value, Symbolic}
2. No true single-graph: Each shape combination still needs its own compiled graph
3. Bucketing is the key optimization: Reduces unique shapes, not eliminates them
4. SimpleGo doesn't need specialization: It already handles any shape at runtime
5. XLA remains shape-dependent: This is fundamental to JIT compilation
6. Pattern caching as pre-filter: Exact matching remains fast path

---
Success Metrics

1. Backward compatible: All existing tests pass unchanged
2. Memory efficiency: Reduced cache bloat with bucketing (10x fewer graphs for variable batches)
3. Developer experience: Shape.WithDynamicBatch() for easy dynamic batch creation
4. Training stability: Gradients work correctly with variable batch sizes
5. Performance: <10% overhead from pattern matching vs exact caching

---
