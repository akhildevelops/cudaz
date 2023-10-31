const std = @import("std");
const testing = std.testing;
const cuda = @import("cuda.zig");
const Error = @import("error.zig");

fn someCuda() !void {
    var device: i32 = 10;
    var dummy: u64 = 10;
    var context: cuda.CUcontext = undefined;
    std.debug.print("{s}\n", .{"start"});
    try Error.fromCudaErrorCode(cuda.cuInit(0));
    std.debug.print("{s}\n", .{"init"});
    try Error.fromCudaErrorCode(cuda.cuDeviceGet(&device, 0));
    std.debug.print("{s}\n", .{"get_device"});
    try Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRetain(&context, device));
    std.debug.print("{s}\n", .{"get_context"});
    try Error.fromCudaErrorCode(cuda.cuCtxSetCurrent(context));
    std.debug.print("{s}\n", .{"set_context"});
    try Error.fromCudaErrorCode(cuda.cuMemAlloc_v2(&dummy, 1024 * 1024 * 1024 * 5));
    std.debug.print("{s}\n", .{"allocated"});
    std.time.sleep(10 * std.time.ns_per_s);
}

test "basic add functionality" {
    try testing.expectError(error.fhfhd, someCuda());
    // try testing.expect(some_cuda() == 0);

    // const cuda_err = error_z.cuda_errors_to_error();
    // _ = cuda_err;
    // std.debug.print("{d}\n\n", .{2});
}
