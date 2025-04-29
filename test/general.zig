const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;
const std = @import("std");

const time = std.time;
test "Setup" {
    const device = try CuDevice.default();
    defer device.deinit();
}

test "Allocate" {
    const device = try CuDevice.default();
    defer device.deinit();
    const slice = try device.alloc(f32, 1024 * 1024 * 1024);
    defer slice.free();
}

test "host_to_device" {
    const device = try CuDevice.default();
    defer device.deinit();
    const slice = try device.htodCopy(f32, &[_]f32{ 1.2, 3.4, 8.9 });
    defer slice.free();
}

test "device_to_host" {
    const device = try CuDevice.default();
    defer device.deinit();
    var float_arr = [_]f32{ 1.2, 3.4, 8.9 };
    const slice = try device.htodCopy(f32, &float_arr);
    var arr = try CuDevice.syncReclaim(f32, std.testing.allocator, slice);
    defer arr.deinit();
    try std.testing.expect(std.mem.eql(f32, &float_arr, arr.items));
}
const increment_kernel =
    \\extern "C" __global__ void increment(float *out)
    \\{
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    out[i] = out[i] + 1;
    \\}
;
test "inc_file" {
    const device = try CuDevice.default();
    defer device.deinit();
    const data = [_]f32{ 1.2, 2.8, 0.123 };
    const cu_slice = try device.htodCopy(f32, &data);
    defer cu_slice.free();
    const ptx = try CuCompile.cudaText(increment_kernel, .{}, std.testing.allocator);
    defer std.testing.allocator.free(ptx);
    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("increment");

    try function.run(.{&cu_slice.device_ptr}, CuLaunchConfig{ .block_dim = .{ 3, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });

    const incremented_arr = try CuDevice.syncReclaim(f32, std.testing.allocator, cu_slice);
    defer incremented_arr.deinit();
    for (incremented_arr.items, data) |id, d| {
        std.debug.assert(id - d == 1);
    }
}
// Got segmentation fault because n was declared a  comptime_int construct and after compiling there will be no n, therefore cuda won't able to fetch the value n.
// Even if n is declared as const param there's seg fault, unless it's declared as var
test "ptx_sin_file" {

    // CuDevice Initialization
    var device = try CuDevice.default();
    defer device.deinit();

    //Load module from ptx
    const module = try CuDevice.loadPtx(.{ .raw_path = "cuda/sin.ptx" });
    const func = try module.getFunc("sin_kernel");

    //init variables
    var float_arr = [_]f32{ 1.57, 0.5236, 0, 1.05 };
    const src_slice = try device.htodCopy(f32, &float_arr);
    var dest_slice = try src_slice.clone();

    // Launch config
    const cfg = CuLaunchConfig.for_num_elems(float_arr.len);
    var n = float_arr.len;

    // Run the func
    try func.run(.{ &dest_slice.device_ptr, &src_slice.device_ptr, &n }, cfg);

    const sin_d = try CuDevice.syncReclaim(f32, std.testing.allocator, dest_slice);
    defer sin_d.deinit();

    for (0..float_arr.len) |index| {
        try std.testing.expect(std.math.approxEqAbs(f32, @sin(float_arr[index]), sin_d.items[index], std.math.floatEps(f32)));
    }
}

test "compile_ptx" {
    const file = try std.fs.cwd().openFile("cuda/sin.cu", .{});
    const ptx_data = try CuCompile.cudaFile(file, .{ .use_fast_math = true }, std.testing.allocator);
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

    var device = try CuDevice.default();
    defer device.deinit();

    const module = try CuDevice.loadPtxText(ptx_data);
    const func = try module.getFunc("matmul");

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const a_slice = try device.htodCopy(f32, &a);
    const b_slice = try a_slice.clone();
    var c_slice = try a_slice.clone();

    const cfg = Cuda.LaunchConfig{ .block_dim = .{ 2, 2, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 };
    var n: i32 = 2;

    try func.run(.{ &a_slice.device_ptr, &b_slice.device_ptr, &c_slice.device_ptr, &n }, cfg);

    const arr = try CuDevice.syncReclaim(f32, std.testing.allocator, c_slice);
    defer arr.deinit();

    for (arr.items, [_]f32{ 7.0, 10.0, 15.0, 22.0 }) |c, o| {
        try std.testing.expect(std.math.approxEqAbs(f32, c, o, std.math.floatEps(f32)));
    }
}

// test "runtime_init" {
//     const device = try CuDevice.default();
//     defer device.deinit();
//     const slice = try device.allocR(Cuda.DType.f16, 10);
//     slice.free();
// }
