![AI Generated](cuda_zig.jpeg)
# Cuda library for Zig
This library helps to interact with NVIDIA GPUs from zig. Provides high level interface to communicate with GPU. It can detect cuda installation and link to a project's binary on Linux/MacOS. Check [Customization](https://github.com/akhildevelops/cudaz/tree/main#Customization) to give cuda manual path.


## The library provides below features:
- Memory Allocation in GPU with defined size.
- Copying data from host to gpu and viceversa.
- Compiling (.cu) and loading kernels (.ptx) both from file and text.
- Running kernels with grid/blocks/threads configuration.
- [Generate random numbers](test/rng.zig)

Check [test](./test) folder for code samples.

>Scroll below to go through an example of incrementing each value in an array parallely using GPU.

### Install
Download and save the library path in `build.zig.zon` file by running

#### zig 0.14.0
`zig fetch --save https://github.com/akhildevelops/cudaz/archive/0.2.0.tar.gz`

#### zig 0.13.0
`zig fetch --save https://github.com/akhildevelops/cudaz/archive/0.1.0.tar.gz`


Add cudaz module in your project's `build.zig` file that will link to your project's binary.
```zig
//build.zig
const std = @import("std");

pub fn build(b: *std.Build) !void {
    // exe points to main.zig that uses cudaz
    const exe = b.addExecutable(.{ .name = "main", .root_source_file = .{ .path = "src/main.zig" }, .target = b.host });

    // Point to cudaz dependency
    const cudaz_dep = b.dependency("cudaz", .{});

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.root_module.addImport("cudaz", cudaz_module);

    // Dynamically link to libc, cuda, nvrtc
    exe.linkLibC();
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("nvrtc");

    // Run binary
    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}


```

### Increment Array using GPU
```zig
// src/main.zig

const std = @import("std");
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;

// Cuda Kernel
const increment_kernel =
    \\extern "C" __global__ void increment(float *out)
    \\{
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    out[i] = out[i] + 1;
    \\}
;

pub fn main() !void {
    // Initialize allocator
    var GP = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = GP.deinit();
    const allocator = GP.allocator();

    // Initialize GPU
    const device = try CuDevice.default();
    defer device.deinit();

    // Copy data from host to GPU
    const data = [_]f32{ 1.2, 2.8, 0.123 };
    const cu_slice = try device.htodCopy(f32, &data);
    defer cu_slice.free();

    // Compile and load the Kernel
    const ptx = try CuCompile.cudaText(increment_kernel, .{}, allocator);
    defer allocator.free(ptx);
    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("increment");

    // Run the kernel on the data
    try function.run(.{&cu_slice.device_ptr}, CuLaunchConfig{ .block_dim = .{ 3, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });

    // Retrieve incremented data back to the system
    const incremented_arr = try CuDevice.syncReclaim(f32, allocator, cu_slice);
    defer incremented_arr.deinit();
}
```
For running above code system refer to the example project: [increment](./example/increment)

## Examples:
- [Incrementing array in GPU](example/increment/)
- [Sending Custom Types to GPU](example/custom_type/)

## Customization
- It is intelligent to identify and link to installed cuda libraries. If needed, provide cuda installation path manually by mentioning build parameter `zig build -DCUDA_PATH=<cuda_folder>`.

Inspired from Rust Cuda library: https://github.com/coreylowman/cudarc/tree/main
