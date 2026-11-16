const std = @import("std");
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;
const build_options = @import("build_options");

// CUDA equivalent of the Zig SPIR-V kernel from issue #15. cudaz compiles
// `.cu` sources with NVRTC, so the issue's Zig kernel cannot be loaded as-is.
const mm_kernel = @embedFile("./matmul.cu");

// PTX generated at build time from MMmult.zig by Zig's NVPTX backend.
const mm_ptx_from_zig = @embedFile("MMmult_ptx");

// C = A * B
// A: x rows, y columns
// B: y rows, z columns
// C: x rows, z columns
const X: u32 = 4;
const Y: u32 = 4;
const Z: u32 = 4;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const device = try CuDevice.default();
    defer device.deinit();

    const a = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    const b = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    const d_a = try device.htodCopy(f32, &a);
    defer d_a.free();
    const d_b = try device.htodCopy(f32, &b);
    defer d_b.free();
    const d_c = try device.alloc(f32, X * Z);
    defer d_c.free();

    // Two ways to get PTX:
    // - NVRTC compiles the equivalent CUDA kernel at runtime (default)
    // - `-Dzig_ptx=true` uses PTX generated from the original Zig kernel at
    //   build time (`zig build-obj -target nvptx64-freestanding ... -femit-asm`)
    const ptx: [:0]const u8 = if (build_options.zig_ptx)
        mm_ptx_from_zig
    else
        try CuCompile.cudaText(mm_kernel, .{}, allocator);
    defer if (!build_options.zig_ptx) allocator.free(ptx);

    if (build_options.zig_ptx) {
        std.debug.print("Using PTX generated from the original Zig kernel\n", .{});
    } else {
        std.debug.print("Using PTX generated with NVRTC from matmul.cu\n", .{});
    }

    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("MMmult");

    const launch_config = CuLaunchConfig{
        .block_dim = .{ X * Z, 1, 1 },
        .grid_dim = .{ 1, 1, 1 },
        .shared_mem_bytes = 0,
    };

    if (build_options.repro) {
        // This is how issue #15 calls the kernel: find more details below.
        //
        // `cuLaunchKernel` expects kernelParams to be an array of *pointers*
        // to argument values. Passing `x`, `y`, `z` by value makes CUDA treat
        // the value 4 as a pointer and dereference address 0x4.
        std.debug.print(
            "Reproducing issue #15: scalar args passed by value (expect an invalid launch; SIGSEGV at 0x4 on the reporter's driver)\n",
            .{},
        );
        try function.run(.{
            @as(c_ulong, X),
            @as(c_ulong, Y),
            @as(c_ulong, Z),
            &d_a.device_ptr,
            &d_b.device_ptr,
            &d_c.device_ptr,
        }, launch_config);
    } else {
        // Fixed call: every kernel argument is passed as a pointer to its
        // value. `&d_*.device_ptr` is already a pointer to a CUdeviceptr, so
        // only the three scalars needed local variables.
        var x_arg = X;
        var y_arg = Y;
        var z_arg = Z;
        try function.run(.{
            &x_arg,
            &y_arg,
            &z_arg,
            &d_a.device_ptr,
            &d_b.device_ptr,
            &d_c.device_ptr,
        }, launch_config);
    }

    var result = try CuDevice.syncReclaim(f32, allocator, d_c);
    defer result.deinit(allocator);

    // CPU reference implementation for verification.
    var expected: [X * Z]f32 = @splat(0);
    for (0..X) |row| {
        for (0..Z) |col| {
            var sum: f32 = 0;
            for (0..Y) |i| {
                sum += a[row * Y + i] * b[i * Z + col];
            }
            expected[row * Z + col] = sum;
        }
    }

    std.debug.print("GPU result: {any}\n", .{result.items});
    std.debug.print("CPU result: {any}\n", .{expected});

    for (result.items, expected) |got, want| {
        if (got != want) {
            std.debug.print("FAIL: expected {d}, got {d}\n", .{ want, got });
            return error.WrongResult;
        }
    }
    std.debug.print("PASS: matrix multiplication matches the CPU reference\n", .{});
}
