const std = @import("std");
const testing = std.testing;
const cuda = @import("cuda.zig");
const Error = @import("error.zig");
const CudaError = Error.CudaError;
// fn someCuda() !void {
//     var device: i32 = 10;
//     var dummy: u64 = 10;
//     var context: cuda.CUcontext = undefined;
//     std.debug.print("{s}\n", .{"start"});
//     try Error.fromCudaErrorCode(cuda.cuInit(0));
//     std.debug.print("{s}\n", .{"init"});
//     try Error.fromCudaErrorCode(cuda.cuDeviceGet(&device, 0));
//     std.debug.print("{s}\n", .{"get_device"});
//     try Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRetain(&context, device));
//     std.debug.print("{s}\n", .{"get_context"});
//     try Error.fromCudaErrorCode(cuda.cuCtxSetCurrent(context));
//     std.debug.print("{s}\n", .{"set_context"});
//     try Error.fromCudaErrorCode(cuda.cuMemAlloc_v2(&dummy, 1024 * 1024 * 5));
//     std.debug.print("{s}\n", .{"allocated"});
//     std.time.sleep(10 * std.time.ns_per_s);
// }

// test "basic add functionality" {
//     try testing.expectError(error.CUDA_ERROR_INVALID_VALUE, someCuda());
// }
pub const CudaDevice = struct {
    device: cuda.CUdevice,
    primary_context: cuda.CUcontext,
    ordinal: u16,
    const Self = @This();

    pub fn new(gpu: u16) CudaError.Error!Self {
        var device: cuda.CUdevice = undefined;
        var context: cuda.CUcontext = undefined;
        try Error.fromCudaErrorCode(cuda.cuInit(0));
        try Error.fromCudaErrorCode(cuda.cuDeviceGet(&device, gpu));
        try Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRetain(&context, device));
        try Error.fromCudaErrorCode(cuda.cuCtxSetCurrent(context));
        return .{
            .device = device,
            .primary_context = context,
            .ordinal = gpu,
        };
    }

    pub fn default() CudaError.Error!Self {
        return Self.new(0);
    }

    pub fn alloc(self: Self, comptime data_type: type, length: usize) CudaError.Error!CudaSlice {
        var device_ptr: cuda.CUdeviceptr = undefined;
        try Error.fromCudaErrorCode(cuda.cuMemAlloc(&device_ptr, @sizeOf(data_type) * length));
        return .{ .device_ptr = device_ptr, .len = length, .device = self };
    }
};

const CudaSlice = struct {
    device_ptr: cuda.CUdeviceptr,
    len: usize,
    device: CudaDevice,
    pub fn free(self: @This()) CudaError.Error!void {
        try Error.fromCudaErrorCode(cuda.cuMemFree_v2(self.device_ptr));
    }
};
