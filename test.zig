const all_cuda = @import("src/lib.zig");
const std = @import("std");
test "Setup" {
    const device = try all_cuda.CudaDevice.default();
    const cuda_slice = try device.alloc(f32, 1024 * 1024 * 1024);
    std.time.sleep(10 * std.time.ns_per_s);
    try cuda_slice.free();
}
