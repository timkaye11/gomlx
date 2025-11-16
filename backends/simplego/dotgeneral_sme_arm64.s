//go:build !noasm && arm64

// Package sme assembly implementations
// SME-accelerated operations for Apple M4 and later
// Vectorized version using SVE in streaming mode

#include "textflag.h"

// func dotProduct_sme_asm(a, b unsafe.Pointer, n int64) float32
TEXT ·dotProduct_sme_asm(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0       // R0 = a pointer
	MOVD b+8(FP), R1       // R1 = b pointer
	MOVD n+16(FP), R2      // R2 = n (count)

	// Zero out result location on stack
	MOVW $0, 12(RSP)
	ADD  $12, RSP, R10     // R10 = &result

	// SME streaming mode with SVE vectors
	WORD $0xd503477f       // smstart
	WORD $0x04a0e3e3       // cntw x3 (get vector length in words)
	WORD $0x1e2703e0       // fmov s0, wzr (zero scalar accumulator)
	WORD $0x25b8c002       // mov z2.s, #0 (zero vector accumulator)
	WORD $0x2598e3e0       // ptrue p0.s (all lanes active)

	// Calculate vector count and remainder using actual vector length
	// vec_count = n / vl, remainder = n % vl
	MOVD R2, R9            // R9 = n (copy for division)
	UDIV R3, R9, R9        // R9 = vec_count = n / vl
	MUL  R3, R9, R8        // R8 = vec_count * vl
	SUB  R8, R2, R8        // R8 = remainder = n - (vec_count * vl)

	// Check if we have any full vectors
	CBZ  R9, remainder     // Skip vector loop if vec_count == 0

	// Simple vector loop - just use one accumulator
	MOVD R9, R4            // R4 = vec_count

vectorloop:
	WORD $0xa540a000       // ld1w {z0.s}, p0/z, [x0]
	WORD $0xa540a021       // ld1w {z1.s}, p0/z, [x1]
	WORD $0x65a10002       // fmla z2.s, p0/m, z0.s, z1.s
	ADD  R3<<2, R0, R0
	ADD  R3<<2, R1, R1
	SUBS $1, R4, R4
	BNE  vectorloop

reduce:
	// Reduce accumulator to scalar
	WORD $0x65802040       // faddv s0, p0, z2.s (reduce z2 → s0)

remainder:
	// Handle remainder elements (scalar loop)
	CBZ  R8, done          // Skip if no remainder

	MOVD R8, R4            // R4 = remainder count
scalarloop:
	WORD $0xbc404403       // ldr s3, [x0], #4  (load a[i])
	WORD $0xbc404424       // ldr s4, [x1], #4  (load b[i])
	WORD $0x1e240865       // fmul s5, s3, s4   (multiply)
	WORD $0x1e252800       // fadd s0, s0, s5   (accumulate)
	SUBS $1, R4, R4        // decrement
	BNE  scalarloop        // loop

done:
	// Store result BEFORE smstop
	WORD $0xbd000140       // str s0, [x10]
	WORD $0xd503467f       // smstop

	// Load from stack and return
	WORD $0xbd400fe0       // ldr s0, [sp, #0xc]
	FMOVS F0, ret+24(FP)   // Store to return value
	RET
