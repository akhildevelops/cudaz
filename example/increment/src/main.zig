const std = @import("std");
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;

// Cuda Kernel
const increment_kernel = @embedFile("./increment.cu");

pub fn main() !void {
    // Initialize allocator
    var GP = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = GP.deinit();
    const allocator = GP.allocator();
    std.debug.print("Initialized allocator\n", .{});
    // Initialize GPU
    const device = try CuDevice.default();
    defer device.deinit();
    std.debug.print("Cuda device is setup\n", .{});

    // Copy data from host to GPU
    const data = [_]f32{ 1.2, 2.8, 0.123 };
    const cu_slice = try device.htodCopy(f32, &data);
    defer cu_slice.free();
    std.debug.print("Copied array {d:.3} from system to GPU\n", .{data});

    // Compile and load the Kernel
    std.debug.print("Kernel program:\n{s}\n\n", .{increment_kernel});
    const ptx = try CuCompile.cudaText(increment_kernel, .{}, allocator);
    defer allocator.free(ptx);
    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("increment");
    std.debug.print("Compiled Cuda Kernel that increments each value by 1 and loaded into GPU\n", .{});

    // Run the kernel on the data
    try function.run(.{&cu_slice.device_ptr}, CuLaunchConfig{ .block_dim = .{ 3, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
    std.debug.print("Ran the Kernel against the array in GPU\n", .{});

    // Retrieve incremented data back to the system
    const incremented_arr = try CuDevice.syncReclaim(f32, allocator, cu_slice);
    defer incremented_arr.deinit();
    std.debug.print("Retrieved incremented data {d:.3} from GPU to system\n", .{incremented_arr.items});
}
