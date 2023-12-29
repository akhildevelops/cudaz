# A Zig Cuda wrapper
### Works with latest zig nightly
This library helps to interact with NVIDIA GPUs from zig. Provides high level interface to communicate with GPU. It can detect cuda installation and link to a project's binary.

>Scroll below to go through an example of incrementing each value in an array parallely using GPU.

### Install

Download and save the library path in `build.zig.zon` file by running

`zig fetch --save https://github.com/akhildevelops/cudaz/tarball/master`

Add cudaz module in your project's `build.zig` file that will link to your project's binary.
```zig
const std = @import("std");

pub fn build(b: *std.Build) !void {
    /////////////////////////////////////////////////////
    // exe points to main.zig that uses cudaz module from the library
    const exe = b.addExecutable(.{ .name = "main", .root_source_file = .{ .path = "src/main.zig" } });

    ////////////////////////////////////////////////////
    // Add cudaz module, include cuda header files and libraries.

    // Point to cudaz dependency
    const cudaz_dep = b.dependency("cudaz", .{});

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.addModule("cudaz", cudaz_module);

    // Point to cudaz library and add obtained header files and library paths to exe.
    const cudaz_lib = cudaz_dep.artifact("cudaz");
    exe.addIncludePath(cudaz_lib.include_dirs.items[0].path);
    for (cudaz_lib.lib_paths.items) |path| {
        exe.addLibraryPath(path);
    }

    // Dynamically link to libc, cuda, nvrtc
    exe.linkLibC();
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("nvrtc");

    //////////////////////////////////////////////////
    // Install binary in zig-out/bin
    b.installArtifact(exe);
}

```

### Increment Array using GPU
```zig
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
    defer GP.deinit();
    const allocator = GP.allocator();

    // Initialize GPU
    const device = try CuDevice.default();
    defer device.free();

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
## Customization
- It is intelligent to identify and link to installed cuda libraries. If needed, provide cuda installation path manually by mentioning build parameter `zig build -DCUDA_PATH=<cuda_folder>`.


## The library provides below features:
- Memory Allocation in GPU with defined size.
- Copying data from host to gpu and viceversa.
- Compiling (.cu) and loading kernels (.ptx) both from file and text.
- Running kernels with grid/blocks/threads configuration.

Check [test.zig](./test.zig) file for code samples.

Inspired from Rust Cuda library: https://github.com/coreylowman/cudarc/tree/main