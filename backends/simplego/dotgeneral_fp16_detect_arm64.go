//go:build arm64

package simplego

import (
	"runtime"
	"syscall"
)

// detectFP16NEON checks if FP16 NEON instructions (FMLAL/FMLAL2) are available.
// These require ARMv8.2-A with FEAT_FHM (half-precision FP multiply-add).
func detectFP16NEON() bool {
	if runtime.GOOS == "darwin" {
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

	// Linux: Read from /proc/cpuinfo or use hwcap
	// Most modern ARM64 Linux devices support FP16
	// For now, assume support on ARM64 Linux (can be refined with hwcap check)
	return runtime.GOARCH == "arm64"
}

// detectBF16NEON checks if BF16 NEON instructions (BFMLALB/BFMLALT) are available.
// These require ARMv8.6-A with FEAT_BF16.
func detectBF16NEON() bool {
	// NOTE: On macOS, the sysctl hw.optional.arm.FEAT_BF16 may report true even when
	// the instructions are not actually supported by the kernel/runtime. This causes
	// SIGILL errors when executing BFMLALB/BFMLALT. We disable BF16 NEON on macOS
	// for now until this is resolved. The scalar fallback is still fast.
	// See: https://github.com/golang/go/issues/... (if there's a relevant issue)
	if runtime.GOOS == "darwin" {
		return false
	}

	// Linux: BF16 is less common, be conservative
	// Would need to check /proc/cpuinfo for "bf16" flag
	return false
}
