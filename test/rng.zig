const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuRng = Cuda.Rng;
const std = @import("std");

test "curand" {
    const curand = try CuRng.default();
    const slice = try curand.genrandom(100);
    defer slice.free();
    const arr = try CuDevice.syncReclaim(f32, std.testing.allocator, slice);
    defer arr.deinit();
}
