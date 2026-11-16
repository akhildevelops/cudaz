# Issue #15: matrix multiplication

Reproduction workspace for
[issue #15](https://github.com/akhildevelops/cudaz/issues/15).

The issue contains three separate problems:

1. **The GPU code is a Zig SPIR-V kernel, not CUDA.** It uses
   `callconv(.kernel)`, `addrspace(.global)`, `@workGroupId`,
   `@workGroupSize`, and `@workItemId`. cudaz compiles `.cu` files with NVRTC
   and loads PTX/cubin with `cuModuleLoadData`, so CUDA cannot load that
   SPIR-V file directly. The original `MMmult.zig` is kept in `src/` as a
   reference. It does compile:

   ```bash
   zig build-obj src/MMmult.zig -target spirv64-vulkan
   ```

   but passing the resulting `.spv` to `cuModuleLoadData` returns
   `CUDA_ERROR_INVALID_IMAGE` (200), so CUDA rejects it. The same Zig kernel
   *can* be compiled to PTX with Zig's LLVM NVPTX backend (see below), or
   replaced by the CUDA equivalent in `src/matmul.cu`.
2. **The kernel is launched with scalar arguments passed by value.**
   `cuLaunchKernel` expects an array of pointers to argument values.
   `&d_A.device_ptr` is correct, but `x`, `y`, `z` must also be passed as
   pointers (`&x`, `&y`, `&z`). Passing the value `4` makes CUDA dereference
   address `0x4`, which is the reported segfault.
3. **The index math is also dubious.** For a row-major `C` with `x` rows and
   `z` columns, `row` and `col` should both use `z` (`gid / z`, `gid % z`).
   The original mixes `gid % x` with division by `z`, so it only works when
   `x == z`. Its `A`/`B` indexing also computes `B * A` rather than the usual
   row-major `A * B`. `src/matmul.cu` uses the standard definition
   (`A`: x×y, `B`: y×z, `C`: x×z).

## Run the fixed version

```bash
zig build run
```

This compiles `src/matmul.cu` with NVRTC at runtime.

Expected result:

```text
PASS: matrix multiplication matches the CPU reference
```

## Compile the original Zig kernel to PTX

Zig can emit PTX through LLVM's NVPTX backend:

```bash
zig build-obj src/MMmult.zig \
  -target nvptx64-freestanding \
  -O ReleaseFast \
  -fno-emit-bin \
  -femit-asm=MMmult.ptx
```

`-fno-emit-bin` is required. Without it Zig reports:

```text
error: cannot emit nvptx64 binary with the LLVM backend; only '-femit-asm' is supported
```

The generated file is normal PTX and can be loaded with
`cuModuleLoadData` / `Device.loadPtxText`. `build.zig` wires the same command
into the build, and the result is embedded as `MMmult_ptx`:

```bash
zig build run -Dzig_ptx=true
```

Expected result:

```text
Using PTX generated from the original Zig kernel
PASS: matrix multiplication matches the CPU reference
```

Note: the original kernel's indexing effectively computes `B * A`; this
example uses `A == B`, so the reference check still matches. `src/matmul.cu`
uses the usual row-major `A * B`.

## Reproduce the original segfault

```bash
zig build run -Drepro=true
```

On the current CUDA driver this exits with:

```text
error: CUDA_ERROR_INVALID_VALUE
```

On the reporter's driver it crashed with `Segmentation fault at address 0x4`.
Either way, the cause is the same: the scalar arguments are not pointers.
The only difference between this and the working path is the launch
parameters; the kernel and the rest of the setup are identical.
