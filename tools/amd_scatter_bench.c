// amd_scatter_bench.c: times the memory patterns of the Round kernels on an
// AMD GPU through the ROCm OpenCL runtime, which compiles the kernels itself,
// so it needs no hipcc. Results and what they mean: docs/amd_results_2026-10-28.md.
//
//   gcc -O2 -o amd_scatter_bench tools/amd_scatter_bench.c -l:libOpenCL.so.1 && ./amd_scatter_bench
//
// The OpenCL prototypes are declared here because the ROCm runtime packages
// ship no CL headers.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong; typedef void* P;
extern cl_int clGetPlatformIDs(cl_uint, P*, cl_uint*);
extern cl_int clGetDeviceIDs(P, cl_ulong, cl_uint, P*, cl_uint*);
extern P clCreateContext(const intptr_t*, cl_uint, const P*, P, P, cl_int*);
extern P clCreateCommandQueueWithProperties(P, P, const cl_ulong*, cl_int*);
extern P clCreateBuffer(P, cl_ulong, size_t, P, cl_int*);
extern P clCreateProgramWithSource(P, cl_uint, const char**, const size_t*, cl_int*);
extern cl_int clBuildProgram(P, cl_uint, const P*, const char*, P, P);
extern cl_int clGetProgramBuildInfo(P, P, cl_uint, size_t, P, size_t*);
extern P clCreateKernel(P, const char*, cl_int*);
extern cl_int clSetKernelArg(P, cl_uint, size_t, const void*);
extern cl_int clEnqueueNDRangeKernel(P, P, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const P*, P*);
extern cl_int clEnqueueFillBuffer(P, P, const void*, size_t, size_t, size_t, cl_uint, const P*, P*);
extern cl_int clFinish(P);
extern cl_int clGetEventProfilingInfo(P, cl_uint, size_t, P, size_t*);
extern cl_int clGetDeviceInfo(P, cl_uint, size_t, P, size_t*);

static const char *src_cl =
"#define NB 16384u\n#define MAXOUT 11264u\n"
"__kernel void copy_coalesced(__global const uint2 *s, __global uint2 *d, uint n) {\n"
"  uint i = get_global_id(0); if (i < n) d[i] = s[i]; }\n"
"__kernel void scatter_atomic(__global const uint2 *s, __global uint2 *d, __global uint *idx, uint n) {\n"
"  uint i = get_global_id(0); if (i >= n) return; uint2 e = s[i]; uint b = e.x >> 18;\n"
"  uint slot = atomic_inc(idx + b); d[b * MAXOUT + min(slot, MAXOUT - 1)] = e; }\n"
"__kernel void scatter_noatomic(__global const uint2 *s, __global uint2 *d, __global uint *idx, uint n) {\n"
"  uint i = get_global_id(0); if (i >= n) return; uint2 e = s[i]; uint b = e.x >> 18;\n"
"  d[b * MAXOUT + (i % MAXOUT)] = e; }\n"
"__kernel void atomic_only(__global const uint2 *s, __global uint2 *d, __global uint *idx, uint n) {\n"
"  uint i = get_global_id(0); if (i >= n) return; uint2 e = s[i]; uint b = e.x >> 18;\n"
"  uint slot = atomic_inc(idx + b); if (slot == 0xfffffffeu) d[0] = e; }\n"
"__kernel void staged_lds32(__global const uint2 *s, __global uint2 *d, __global uint *idx, uint n) {\n"
"  /* same scatter, but each lane writes 32 B (4 edges) to its bucket: what an LDS-staged flush would issue */\n"
"  uint i = get_global_id(0); if (i * 4 >= n) return; uint2 e = s[i * 4]; uint b = e.x >> 18;\n"
"  uint slot = atomic_add(idx + b, 4u); slot = min(slot, MAXOUT - 4) & ~3u;\n"
"  __global uint4 *dd = (__global uint4 *)(d + b * MAXOUT + slot);\n"
"  dd[0] = (uint4)(e.x, e.y, s[i*4+1].x, s[i*4+1].y); dd[1] = (uint4)(s[i*4+2].x, s[i*4+2].y, s[i*4+3].x, s[i*4+3].y); }\n"
"__kernel void scatter128_8B(__global const uint2 *s, __global uint2 *d, __global uint *idx, uint n) {\n"
"  /* first level of a two-level round: 128 destinations, plain 8 B per lane */\n"
"  uint i = get_global_id(0); if (i >= n) return; uint2 e = s[i]; uint b = e.x >> 25;\n"
"  uint slot = atomic_inc(idx + b); d[(size_t)b * (n / 100) + (slot % (n / 100))] = e; }\n"
"__kernel void scatter128_32B(__global const uint2 *s, __global uint2 *d, __global uint *idx, uint n) {\n"
"  uint i = get_global_id(0); if (i * 4 >= n) return; uint2 e = s[i * 4]; uint b = e.x >> 25;\n"
"  uint slot = atomic_add(idx + b, 4u) % (n / 100); slot &= ~3u; __global uint4 *dd = (__global uint4 *)(d + (size_t)b * (n / 100) + slot);\n"
"  dd[0] = (uint4)(e.x, e.y, s[i*4+1].x, s[i*4+1].y); dd[1] = (uint4)(s[i*4+2].x, s[i*4+2].y, s[i*4+3].x, s[i*4+3].y); }\n"
"__kernel __attribute__((reqd_work_group_size(1024,1,1))) void lds_count(__global const uint2 *s, __global uint *out, uint per_block) {\n"
"  __local uint cnt[2049]; uint lid = get_local_id(0); uint g = get_group_id(0);\n"
"  for (uint k = lid; k < 2049; k += 1024) cnt[k] = 0; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  __global const uint2 *b = s + (size_t)g * per_block;\n"
"  for (uint k = lid; k < per_block; k += 1024) { uint node = b[k].y & 0x7fffu; uint w = node >> 5, m = 1u << (node & 31);\n"
"    uint old = atomic_or(cnt + w, m) & m; if (old) atomic_or(cnt + w + 1025, m); }\n"
"  barrier(CLK_LOCAL_MEM_FENCE); if (lid == 0) out[g] = cnt[1025]; }\n";

static double ms(P ev) { cl_ulong a, b; clGetEventProfilingInfo(ev, 0x1282, 8, &a, 0); clGetEventProfilingInfo(ev, 0x1283, 8, &b, 0); return (b - a) / 1e6; }

int main() {
  P plat, dev, ctx, q, prog; cl_int err;
  clGetPlatformIDs(1, &plat, 0); clGetDeviceIDs(plat, 4, 1, &dev, 0);
  char name[128]; clGetDeviceInfo(dev, 0x102B, sizeof name, name, 0);
  ctx = clCreateContext(0, 1, &dev, 0, 0, &err);
  cl_ulong props[] = {0x1093, 2, 0}; q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
  prog = clCreateProgramWithSource(ctx, 1, &src_cl, 0, &err);
  if (clBuildProgram(prog, 1, &dev, "", 0, 0)) { char log[8192]; clGetProgramBuildInfo(prog, dev, 0x1183, sizeof log, log, 0); printf("build failed:\n%s\n", log); return 1; }
  const uint32_t N = 128u << 20; // 128M edges, 1 GB; Round 0 reads 512M and writes 324M
  uint32_t *h = malloc((size_t)N * 8); uint64_t x = 88172645463325252ull;
  for (size_t k = 0; k < (size_t)N * 2; k++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; h[k] = (uint32_t)x; }
  P s = clCreateBuffer(ctx, 1 | 32, (size_t)N * 8, h, &err);
  P d = clCreateBuffer(ctx, 1, (size_t)16384 * 11264 * 8, 0, &err);
  P idx = clCreateBuffer(ctx, 1, 16384 * 4 + 4096 * 4, 0, &err);
  printf("%s, %u M edges per kernel\n", name, N >> 20);
  const char *names[] = {"copy_coalesced", "scatter_atomic", "scatter_noatomic", "atomic_only", "staged_lds32", "lds_count", "scatter128_8B", "scatter128_32B"};
  const int per[] = {1,1,1,1,4,1,1,4};
  for (int k = 0; k < 8; k++) {
    P kern = clCreateKernel(prog, names[k], &err); uint32_t zero = 0;
    double best = 1e9;
    for (int rep = 0; rep < 4; rep++) {
      clEnqueueFillBuffer(q, idx, &zero, 4, 0, 16384 * 4 + 4096 * 4, 0, 0, 0);
      size_t gsz, lsz = k == 5 ? 1024 : 256; uint32_t n = N;
      clSetKernelArg(kern, 0, 8, &s);
      if (k == 5) { uint32_t per_block = 32768; clSetKernelArg(kern, 1, 8, &idx); clSetKernelArg(kern, 2, 4, &per_block); gsz = (size_t)(N / 32768) * 1024; }
      else if (k == 0) { clSetKernelArg(kern, 1, 8, &d); clSetKernelArg(kern, 2, 4, &n); gsz = N; }
      else { clSetKernelArg(kern, 1, 8, &d); clSetKernelArg(kern, 2, 8, &idx); clSetKernelArg(kern, 3, 4, &n); gsz = N / per[k]; }
      P ev; err = clEnqueueNDRangeKernel(q, kern, 1, 0, &gsz, &lsz, 0, 0, &ev); clFinish(q);
      if (err) { printf("%s: enqueue error %d\n", names[k], err); break; }
      double t = ms(ev); if (t < best) best = t;
    }
    double bytes = (k == 5) ? N * 8.0 : (k == 3 ? N * 8.0 : N * 16.0);
    printf("%-18s %8.2f ms   %6.0f GB/s moved   %7.1f M edges/ms\n", names[k], best, bytes / best / 1e6, N / best / 1e6 * 1e3 / 1e3);
  }
  return 0;
}
