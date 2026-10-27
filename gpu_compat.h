// gpu_compat.h - builds the CUDA sources with HIP for AMD GPUs.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Under nvcc this header includes nothing: the tari_* warp helpers below
// expand to the CUDA intrinsics they stand for, so the NVIDIA build compiles
// exactly the code it compiled before.
//
// Under the HIP compiler (hipcc, AMD) it includes hip/hip_runtime.h, maps the
// CUDA runtime names the miner uses to their hip* equivalents, and implements
// the warp helpers for 32-lane wavefronts. The AMD build is wave32 only
// (RDNA2, RDNA3 and RDNA4, compiled with -mno-wavefrontsize64): the kernels
// use 32-bit lane masks and `threadIdx.x & 31` lane numbers.
//
// The kernels call tari_* names rather than the CUDA _sync intrinsics, because
// recent HIP versions define __shfl_sync, __ballot_sync and __match_any_sync
// themselves with 64-bit masks, and reject 32-bit ones.

#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define TARI_GPU_HIP 1
#else
#define TARI_GPU_HIP 0
#endif

#if TARI_GPU_HIP

#include <hip/hip_runtime.h>

// hip/amd_detail/amd_device_functions.h leaves these helper macros defined,
// and compress.hpp has a member called MASK2.
#undef MASK1
#undef MASK2

#define TARI_GPU_RUNTIME "HIP"

// Shared memory (LDS) available to one workgroup on RDNA.
#define TARI_HIP_MAX_LDS_BYTES 65536

// The device-side wavefront width is fixed at compile time. Toolchains that
// still define __AMDGCN_WAVEFRONT_SIZE (deprecated in recent clang) are checked
// here; otherwise the runtime warpSize check in create_solver_ctx() is the
// guard.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE)
#if __AMDGCN_WAVEFRONT_SIZE != 32
#error "The AMD build needs 32-lane wavefronts: compile with -mno-wavefrontsize64 for RDNA (gfx10xx, gfx11xx, gfx12xx)"
#endif
#endif
#pragma clang diagnostic pop

// ulonglong4 is 32-byte aligned in HIP. Shared arrays that are read back as
// ulonglong4 get the same alignment.
#define TARI_LDS_ALIGN_ULL4 __attribute__((aligned(32)))

// --- CUDA runtime names used by the miner, mapped to HIP ---
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost tari_hip_malloc_host
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync

#define cudaStream_t hipStream_t
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

#define cudaFuncSetAttribute tari_hip_func_set_attribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// cudaMallocHost(ptr, size): pinned host memory with default flags.
static inline hipError_t tari_hip_malloc_host(void **ptr, size_t size) {
  return hipHostMalloc(ptr, size, 0);
}

// hipFuncSetAttribute takes the kernel as const void *, where CUDA takes the
// kernel itself.
template <typename Kernel>
static inline hipError_t tari_hip_func_set_attribute(Kernel kernel, hipFuncAttribute attr, int value) {
  return hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), attr, value);
}

// --- Warp helpers for 32-lane wavefronts ---
// __ballot returns a 64-bit mask; in wave32 the upper half is zero. As in
// CUDA, every lane named in `mask` must be active and call the helper.

__device__ __forceinline__ unsigned tari_activemask() {
  return (unsigned)__ballot(1);
}

__device__ __forceinline__ unsigned tari_ballot_sync(unsigned mask, int predicate) {
  return (unsigned)__ballot(predicate) & mask;
}

__device__ __forceinline__ unsigned tari_shfl_sync(unsigned mask, unsigned value, int srcLane) {
  (void)mask;
  return __shfl(value, srcLane, 32);
}

// Lanes of `mask` holding the same value as this lane. One ballot per
// distinct value: each pass takes the lowest remaining lane's value and
// removes every lane that holds it. `remaining` is the same in every lane, so
// all lanes run the same number of passes.
//
// That makes it a loop of up to 32 cross-lane round trips, not one
// instruction like CUDA's __match_any_sync. The trimmer therefore defaults
// WARP_DST_ATOMICS to 0 under HIP; this helper only runs when a build turns
// it back on.
__device__ __forceinline__ unsigned tari_match_any_sync(unsigned mask, unsigned value) {
  const unsigned lane = __lane_id();
  unsigned remaining = mask;
  unsigned result = 0;
  while (remaining) {
    const int leader = __ffs(remaining) - 1;
    const unsigned leaderValue = __shfl(value, leader, 32);
    const unsigned same = (unsigned)__ballot(value == leaderValue) & remaining;
    if ((same >> lane) & 1)
      result = same;
    remaining &= ~same;
  }
  return result;
}

#else // nvcc

#define TARI_GPU_RUNTIME "CUDA"
#define TARI_LDS_ALIGN_ULL4

#define tari_activemask() __activemask()
#define tari_ballot_sync(mask, predicate) __ballot_sync(mask, predicate)
#define tari_shfl_sync(mask, value, srcLane) __shfl_sync(mask, value, srcLane)
#define tari_match_any_sync(mask, value) __match_any_sync(mask, value)

#endif
