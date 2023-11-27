const Cuda = @import("src/lib.zig");
const Device = Cuda.Device;
const std = @import("std");
const path = @import("src/path.zig");

const time = std.time;
test "Setup" {
    const device = try Device.default();
    defer device.free();
}

test "Allocate" {
    const device = try Device.default();
    defer device.free();
    const slice = try device.alloc(f32, 1024 * 1024 * 1024);
    defer slice.free();
}

test "host_to_device" {
    const device = try Device.default();
    defer device.free();
    const slice = try device.htod_copy(f32, &[_]f32{ 1.2, 3.4, 8.9 });
    defer slice.free();
}

test "device_to_host" {
    const device = try Device.default();
    defer device.free();
    var float_arr = [_]f32{ 1.2, 3.4, 8.9 };
    const slice = try device.htod_copy(f32, &float_arr);
    var arr = try Device.sync_reclaim(f32, std.testing.allocator, slice);
    defer arr.deinit();
    try std.testing.expect(std.mem.eql(f32, &float_arr, arr.items));
}

// Got segmentation fault because n was declared a  comptime_int construct and after compiling there will be no n, therefore cuda won't able to fetch the value n.
// Even if n is declared as const param there's seg fault, unless it's declared as var
test "ptx_file" {

    // Device Initialization
    var device = try Device.default();
    defer device.free();

    //Load module from ptx
    const module = try Device.load_ptx(.{ .raw_path = "cuda/sin.ptx" });
    const func = try module.get_func("sin_kernel");

    //init variables
    var float_arr = [_]f32{ 1.57, 0.5236, 0, 1.05 };
    const src_slice = try device.htod_copy(f32, &float_arr);
    var dest_slice = try src_slice.clone();

    // Launch config
    const cfg = Cuda.LaunchConfig.for_num_elems(float_arr.len);
    var n = float_arr.len;

    // Run the func
    try func.run(.{ &dest_slice.device_ptr, &src_slice.device_ptr, &n }, cfg);

    const sin_d = try Device.sync_reclaim(f32, std.testing.allocator, dest_slice);
    defer sin_d.deinit();

    for (0..float_arr.len) |index| {
        try std.testing.expect(std.math.approxEqAbs(f32, @sin(float_arr[index]), sin_d.items[index], std.math.floatEps(f32)));
    }
}

test "compile_ptx" {
    const file = try std.fs.cwd().openFile("cuda/sin.cu", .{});
    const ptx_data = try Cuda.Compile.cudaFile(file, .{ .use_fast_math = true }, std.testing.allocator);
    std.testing.allocator.free(ptx_data);
}

const cuda_src =
    \\extern "C" __global__ void matmul(float* A, float* B, float* C, const int N) {
    \\    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    \\    int COL = blockIdx.x*blockDim.x+threadIdx.x;
    \\    
    \\    float tmpSum = 0;
    \\
    \\    if (ROW < N && COL < N) {
    \\        // each thread computes one element of C
    \\        for (int i = 0; i < N; i++) {
    \\            tmpSum += A[ROW * N + i] * B[i * N + COL];
    \\        }
    \\    }
    \\    C[ROW * N + COL] = tmpSum;
    \\ }
;

test "matmul_kernel" {
    const ptx_data = try Cuda.Compile.cudaText(cuda_src, .{}, std.testing.allocator);
    defer std.testing.allocator.free(ptx_data);

    var device = try Cuda.Device.default();
    defer device.free();

    const module = try Cuda.Device.load_ptx_text(ptx_data);
    const func = try module.get_func("matmul");

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const a_slice = try device.htod_copy(f32, &a);
    const b_slice = try a_slice.clone();
    var c_slice = try a_slice.clone();

    const cfg = Cuda.LaunchConfig{ .block_dim = .{ 2, 2, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 };
    var n: i32 = 2;

    try func.run(.{ &a_slice.device_ptr, &b_slice.device_ptr, &c_slice.device_ptr, &n }, cfg);

    const arr = try Device.sync_reclaim(f32, std.testing.allocator, c_slice);
    defer arr.deinit();

    for (arr.items, [_]f32{ 7.0, 10.0, 15.0, 22.0 }) |c, o| {
        try std.testing.expect(std.math.approxEqAbs(f32, c, o, std.math.floatEps(f32)));
    }
}
