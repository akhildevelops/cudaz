const std = @import("std");
const path = @import("path.zig");
const cuda = @import("c.zig").cuda;
const Error = @import("error.zig");
const LaunchConfig = @import("launchconfig.zig").LaunchConfig;
const CudaError = Error.CudaError;
const Module = @import("wrappers.zig").Module;
const CudaFunction = @import("wrappers.zig").Function;
const CudaSlice = @import("wrappers.zig").CudaSlice;
const CudaSliceR = @import("wrappers.zig").CudaSliceR;
const DType = @import("utils.zig").DType;
device: cuda.CUdevice,
primary_context: cuda.CUcontext,
ordinal: u16,
const Device = @This();

pub fn loadPtx(file_path: path.PathBuffer) CudaError.Error!Module {
    var module: cuda.CUmodule = undefined;
    try Error.fromCudaErrorCode(cuda.cuModuleLoad(&module, file_path.raw_path.ptr));
    return .{ .cu_module = module };
}

pub fn loadPtxText(
    ptx: [:0]const u8,
) CudaError.Error!Module {
    var module: cuda.CUmodule = undefined;
    try Error.fromCudaErrorCode(cuda.cuModuleLoadData(&module, ptx.ptr));
    return .{ .cu_module = module };
}

pub fn new(gpu: u16) CudaError.Error!Device {
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

pub fn default() CudaError.Error!Device {
    return Device.new(0);
}

pub fn alloc(self: Device, comptime T: type, length: usize) CudaError.Error!CudaSlice(T) {
    var device_ptr: cuda.CUdeviceptr = undefined;
    try Error.fromCudaErrorCode(cuda.cuMemAlloc(&device_ptr, @sizeOf(T) * length));
    return .{ .device_ptr = device_ptr, .len = length, .device = self };
}
pub fn allocR(self: Device, dtype: DType, length: usize) CudaError.Error!CudaSliceR {
    var device_ptr: cuda.CUdeviceptr = undefined;
    try Error.fromCudaErrorCode(cuda.cuMemAlloc(&device_ptr, dtype.size() * length));
    return .{ .device_ptr = device_ptr, .len = length, .device = self, .element_type = dtype };
}
pub fn allocZeros(self: Device, comptime data_type: type, length: usize) CudaError.Error!CudaSlice(data_type) {
    const slice = try self.alloc(data_type, length);
    try Device.memsetZeros(data_type, slice);
    return slice;
}
pub fn allocZerosR(self: Device, dtype: DType, length: usize) CudaError.Error!CudaSliceR {
    const slice = try self.allocR(dtype, length);
    try Device.memsetZeros(dtype, slice);
    return slice;
}

pub fn memsetZeros(comptime data_type: type, cuda_slice: CudaSlice(data_type)) CudaError.Error!void {
    try Error.fromCudaErrorCode(cuda.cuMemsetD8_v2(cuda_slice.device_ptr, 0, cuda_slice.len * @sizeOf(data_type)));
}

pub fn memsetZerosR(dtype: DType, cuda_slice: CudaSliceR) CudaError.Error!void {
    try Error.fromCudaErrorCode(cuda.cuMemsetD8_v2(cuda_slice.device_ptr, 0, cuda_slice.len * dtype.size()));
}
pub fn free(ptr: cuda.CUdeviceptr) !void {
    try Error.fromCudaErrorCode(cuda.cuMemFree_v2(ptr));
}
pub fn deinit(self: *const Device) void {
    Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRelease(self.device)) catch |err| @panic(@errorName(err));
}
pub fn htodCopyInto(comptime T: type, src: []const T, destination: CudaSlice(T)) CudaError.Error!void {
    std.debug.assert(src.len == destination.len);
    try Error.fromCudaErrorCode(cuda.cuMemcpyHtoD_v2(destination.device_ptr, @ptrCast(src), @sizeOf(T) * src.len));
}
pub fn htodCopy(self: Device, comptime T: type, src: []const T) CudaError.Error!CudaSlice(T) {
    const slice = try self.alloc(T, src.len);
    try Device.htodCopyInto(T, src, slice);
    return slice;
}
pub fn dtohCopyInto(comptime T: type, src: CudaSlice(T), destination: []T) CudaError.Error!void {
    std.debug.assert(src.len == destination.len);
    try Error.fromCudaErrorCode(cuda.cuMemcpyDtoH_v2(@ptrCast(destination), src.device_ptr, @sizeOf(T) * src.len));
}
pub fn dtohCopy(comptime T: type, allocator: std.mem.Allocator, slice: CudaSlice(T)) ![]T {
    const host_buf: []T = try allocator.alloc(T, slice.len);
    try dtohCopyInto(T, slice, host_buf);
    return host_buf;
}
pub fn syncReclaim(comptime T: type, allocator: std.mem.Allocator, slice: CudaSlice(T)) !std.ArrayList(T) {
    var h_Array = try std.ArrayList(T).initCapacity(allocator, slice.len);
    try Error.fromCudaErrorCode(cuda.cuMemcpyDtoH_v2(@ptrCast(h_Array.items), slice.device_ptr, @sizeOf(T) * slice.len));
    // Zig doesn't know the item pointer length, so setting the length of slice.
    h_Array.items.len = slice.len;
    return h_Array;
}
pub fn syncReclaimR(comptime T: type, allocator: std.mem.Allocator, slice: CudaSliceR) !std.ArrayList(T) {
    var h_Array = try std.ArrayList(T).initCapacity(allocator, slice.len);
    try Error.fromCudaErrorCode(cuda.cuMemcpyDtoH_v2(@ptrCast(h_Array.items), slice.device_ptr, @sizeOf(T) * slice.len));
    h_Array.items.len = slice.len;
}
