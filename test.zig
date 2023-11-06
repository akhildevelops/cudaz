const all_cuda = @import("src/lib.zig");
const std = @import("std");
test "Setup" {
    const device = try all_cuda.CudaDevice.default();
    defer device.free();
}

test "Allocate" {
    const device = try all_cuda.CudaDevice.default();
    defer device.free();
    const dlice = try device.alloc(f32, 1024 * 1024 * 1024);
    std.time.sleep(10 * std.time.ns_per_s);
    defer dlice.free();
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
