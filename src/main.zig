const std = @import("std");
const testing = std.testing;
const cuda = @import("cuda.zig");
const Error = @import("error.zig");
export fn some_cuda() c_uint {
    var device: c_int = 10;
    var dummy: c_ulonglong = 10;
    var context: cuda.CUcontext = undefined;
    const init: i32 = @intCast(cuda.cuInit(0));
    _ = init;
    std.debug.print("{s}\n\n", .{"init"});
    const result = cuda.cuDeviceGet(&device, 0);
    std.debug.print("{d}\n\n", .{result});
    const primary_context = cuda.cuDevicePrimaryCtxRetain(&context, device);
    std.debug.print("{d}\n\n", .{primary_context});
    const set_context = cuda.cuCtxSetCurrent(context);
    std.debug.print("{d}\n\n", .{set_context});
    const allocate = cuda.cuMemAlloc_v2(&dummy, 1024 * 1024 * 1024 * 5);
    std.debug.print("{d}\n\n", .{allocate});
    std.time.sleep(10 * std.time.ns_per_s);

    return result;
}

test "basic add functionality" {
    // try testing.expect(some_cuda() == 0);

    // const cuda_err = error_z.cuda_errors_to_error();
    // _ = cuda_err;
    // std.debug.print("{d}\n\n", .{2});
}
