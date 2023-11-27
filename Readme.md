# A Zig Cuda wrapper

This library helps to interact with NVIDIA GPUs from zig. Below is a tiny example for incrementing each value of a zig array parallely on GPU.


```zig
// Cuda Kernel
const increment_kernel =
    \\extern "C" __global__ void increment(float *out)
    \\{
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    out[i] = out[i] + 1;
    \\}
;

// Initialize GPU
const device = try CuDevice.default();
defer device.free();

// Copy data from host to GPU
const data = [_]f32{ 1.2, 2.8, 0.123 };
const cu_slice = try device.htod_copy(f32, &data);
defer cu_slice.free();

// Compile and load the Kernel
const ptx = try CuCompile.cudaText(increment, .{}, std.testing.allocator);
defer std.testing.allocator.free(ptx);
const module = try CuDevice.load_ptx_text(ptx);
const function = try module.get_func("increment");

// Run the kernel on the data
try function.run(.{&cu_slice.device_ptr}, CuLaunchConfig{ .block_dim = .{ 3, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });

// Retrieve incremented data back to the system
const incremented_arr = try CuDevice.sync_reclaim(f32, std.testing.allocator, cu_slice);
defer incremented_arr.deinit();

```

The library provides below features:
- Memory Allocation in GPU with defined size.
- Copying data from host to gpu and viceversa.
- Compiling (.cu) and loading kernels (.ptx) both from file and text.
- Running kernels with grid/blocks/threads configuration.

Inspired from Rust Cuda library: https://github.com/coreylowman/cudarc/tree/main