//go:build darwin && arm64

package simplego

import (
	"syscall"
)

// detectFP16NEON checks if FP16 NEON instructions (FMLAL/FMLAL2) are available.
// These require ARMv8.2-A with FEAT_FHM (half-precision FP multiply-add).
func detectFP16NEON() bool {
	// macOS: Check for FP16 support via sysctl
	// Apple Silicon (M1+) supports FP16 instructions
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_FHM")
	if err == nil && len(val) > 0 && val[0] != 0 {
		return true
	}
	// Fallback: Apple M1 and later always support FP16
	val, err = syscall.Sysctl("hw.optional.neon_fp16")
	if err == nil && len(val) > 0 && val[0] != 0 {
		return true
	}
	// Another fallback for older macOS versions
	val, err = syscall.Sysctl("machdep.cpu.brand_string")
	if err == nil {
		// Apple M-series chips always support FP16
		for i := range val {
			if val[i] == 'M' && i > 0 && val[i-1] == ' ' {
				return true
			}
		}
	}
	return false
}

// detectBF16NEON checks if BF16 NEON instructions (BFMLALB/BFMLALT) are available.
// These require ARMv8.6-A with FEAT_BF16.
func detectBF16NEON() bool {
	// TODO: BF16 NEON instructions execute without SIGILL but produce wrong results
	// on Apple M4. Need to investigate the instruction semantics further.
	// Disabling for now until we can debug the BF16 implementation.
	return false
}
