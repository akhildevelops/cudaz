const std = @import("std");
const Ctype = @cImport(@cInclude("../c/tuple.h"));
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;

// Cuda Kernel
const increment_kernel = @embedFile("offset.cu");

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

    // Initialize host data with a custom data type
    var src_array = try std.ArrayList(Ctype.tuple).initCapacity(allocator, 10);
    defer src_array.deinit();
    for (0..10) |index| {
        try src_array.append(.{ .x = @floatFromInt(index), .y = @as(f32, @floatFromInt(index)) + std.math.pi });
    }

    // Copy data from host to GPU
    const src_cu_slice = try device.htodCopy(Ctype.tuple, src_array.items);
    defer src_cu_slice.free();
    std.debug.print("Copied tuple array {any} from system to GPU\n", .{src_array.items});

    // Allocate 10 f32 in GPU memory
    const dest_cu_slice = try device.alloc(f32, 10);

    // Compile and load the Kernel
    const ptx = try CuCompile.cudaText(increment_kernel, .{}, allocator);
    defer allocator.free(ptx);

    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("offset");
    std.debug.print("Compiled Cuda Kernel that generates offsets between a tuple pair", .{});

    // Run the kernel on the data
    try function.run(.{ &src_cu_slice.device_ptr, &dest_cu_slice.device_ptr }, CuLaunchConfig{ .block_dim = .{ 10, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
    std.debug.print("Ran the Kernel against the array in GPU\n", .{});

    // Retrieve incremented data back to the system
    const incremented_arr = try CuDevice.syncReclaim(f32, allocator, dest_cu_slice);
    defer incremented_arr.deinit();
    std.debug.print("Retrieved offset data {d:.3} from GPU to system\n", .{incremented_arr.items});
}
