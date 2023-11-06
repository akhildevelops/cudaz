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

    pub fn alloc(self: Self, comptime data_type: type, length: usize) CudaError.Error!CudaSlice(data_type) {
        var device_ptr: cuda.CUdeviceptr = undefined;
        try Error.fromCudaErrorCode(cuda.cuMemAlloc(&device_ptr, @sizeOf(data_type) * length));
        return .{ .device_ptr = device_ptr, .len = length, .device = self };
    }
    pub fn alloc_zeros(self: Self, comptime data_type: type, length: usize) CudaError.Error!CudaSlice(data_type) {
        const slice = try self.alloc(data_type, length);
        try Self.memset_zeros(data_type, slice);
        return slice;
    }
    pub fn memset_zeros(comptime data_type: type, cuda_slice: CudaSlice(data_type)) CudaError.Error!void {
        try Error.fromCudaErrorCode(cuda.cuMemsetD8_v2(cuda_slice.device_ptr, 0, cuda_slice.len * @sizeOf(data_type)));
    }
    pub fn free(self: Self) void {
        Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRelease(self.device)) catch |err| @panic(@errorName(err));
    }
    pub fn htod_copy_into(self: Self, comptime T: type, src: []const T, destination: CudaSlice(T)) CudaError.Error!void {
        _ = self;
        try Error.fromCudaErrorCode(cuda.cuMemcpyHtoD_v2(destination.device_ptr, @ptrCast(src), @sizeOf(T) * src.len));
    }
    pub fn htod_copy(self: Self, comptime T: type, src: []const T) CudaError.Error!CudaSlice(T) {
        const slice = try self.alloc(T, src.len);
        try self.htod_copy_into(T, src, slice);
        return slice;
    }
    pub fn sync_reclaim(self: Self, comptime T: type, allocator: std.mem.Allocator, slice: CudaSlice(T)) !std.ArrayList(T) {
        _ = self;
        var h_Array = try std.ArrayList(T).initCapacity(allocator, slice.len);
        try Error.fromCudaErrorCode(cuda.cuMemcpyDtoH_v2(@ptrCast(h_Array.items), slice.device_ptr, @sizeOf(T) * slice.len));
        // Zig doesn't know the item pointer length, so setting the length of slice.
        h_Array.items.len = slice.len;
        return h_Array;
    }
};

fn CudaSlice(comptime T: type) type {
    return struct {
        device_ptr: cuda.CUdeviceptr,
        len: usize,
        device: CudaDevice,
        pub const element_type: type = T;
        pub fn free(self: @This()) void {
            Error.fromCudaErrorCode(cuda.cuMemFree_v2(self.device_ptr)) catch |err| @panic(@errorName(err));
        }
    };
}
