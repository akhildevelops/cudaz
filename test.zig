const all_cuda = @import("src/lib.zig");
const std = @import("std");
const path = @import("src/path.zig");
test "Setup" {
    const device = try all_cuda.CudaDevice.default();
    defer device.free();
}

test "Allocate" {
    const device = try all_cuda.CudaDevice.default();
    defer device.free();
    const slice = try device.alloc(f32, 1024 * 1024 * 1024);
    defer slice.free();
}

test "host_to_device" {
    const device = try all_cuda.CudaDevice.default();
    defer device.free();
    const slice = try device.htod_copy(f32, &[_]f32{ 1.2, 3.4, 8.9 });
    defer slice.free();
}

test "device_to_host" {
    const device = try all_cuda.CudaDevice.default();
    defer device.free();
    var float_arr = [_]f32{ 1.2, 3.4, 8.9 };
    const slice = try device.htod_copy(f32, &float_arr);
    var arr = try device.sync_reclaim(f32, std.testing.allocator, slice);
    defer arr.deinit();
    try std.testing.expect(std.mem.eql(f32, &float_arr, arr.items));
}

// Got segmentation fault because n was declared a  comptime_int construct and after compiling there will be no n, therefore cuda wan't able to fetch the value n.
// Even if n is declared as const param there's seg fault, unless it's declared as var
test "ptx_file" {
    var device = try all_cuda.CudaDevice.default();
    defer device.free();
    var name = [_][]const u8{
        "sin_kernel",
    };
    try device.load_ptx(.{ .raw_path = "./sin.ptx" }, "sin", &name, std.testing.allocator);
    const func = (try device.get_func("sin", "sin_kernel")).?;
    // sin(0), sin(30), sin(60), sin(90)
    var float_arr = [_]f32{ 1.57, 0.5236, 0, 1.05 };
    const src_slice = try device.htod_copy(f32, &float_arr);
    var dest_slice = try src_slice.clone();
    const cfg = all_cuda.LaunchConfig.for_num_elems(float_arr.len);
    var n: i32 = float_arr.len;
    std.debug.print("{d}\n", .{@sizeOf(@TypeOf(n))});
    try device.launch_func(func, cfg, .{ &dest_slice.device_ptr, &src_slice.device_ptr, &n });
    const sin_d = try device.sync_reclaim(f32, std.testing.allocator, dest_slice);
    defer sin_d.deinit();
    std.debug.print("{any}\n", .{sin_d});
}

test "learn" {
    comptime {
        const x = 32;
        _ = x;
    }
    const x: u16 = undefined;
    std.debug.print("{d}\n", .{x});
}
