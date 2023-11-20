const Cuda = @import("src/lib.zig");
const std = @import("std");
const path = @import("src/path.zig");
test "Setup" {
    const device = try Cuda.Device.default();
    defer device.free();
}

test "Allocate" {
    const device = try Cuda.Device.default();
    defer device.free();
    const slice = try device.alloc(f32, 1024 * 1024 * 1024);
    defer slice.free();
}

test "host_to_device" {
    const device = try Cuda.Device.default();
    defer device.free();
    const slice = try device.htod_copy(f32, &[_]f32{ 1.2, 3.4, 8.9 });
    defer slice.free();
}

test "device_to_host" {
    const device = try Cuda.Device.default();
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

    // Device Initialization
    var device = try Cuda.Device.default();
    defer device.free();

    //Load module from ptx
    const module = try Cuda.Device.load_ptx(.{ .raw_path = "./sin.ptx" });
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

    const sin_d = try device.sync_reclaim(f32, std.testing.allocator, dest_slice);
    defer sin_d.deinit();

    for (0..float_arr.len) |index| {
        try std.testing.expect(std.math.approxEqAbs(f32, @sin(float_arr[index]), sin_d.items[index], std.math.floatEps(f32)));
    }
}
