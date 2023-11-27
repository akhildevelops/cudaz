const std = @import("std");
const path = @import("path.zig");
const cuda = @import("c.zig").cuda;
const Error = @import("error.zig");
const LaunchConfig = @import("launchconfig.zig").LaunchConfig;
const CudaError = Error.CudaError;
const Module = @import("wrappers.zig").Module;
const CudaFunction = @import("wrappers.zig").Function;
const CudaSlice = @import("wrappers.zig").CudaSlice;

device: cuda.CUdevice,
primary_context: cuda.CUcontext,
ordinal: u16,
const Device = @This();

pub fn load_ptx(file_path: path.PathBuffer) CudaError.Error!Module {
    var module: cuda.CUmodule = undefined;
    try Error.fromCudaErrorCode(cuda.cuModuleLoad(&module, file_path.raw_path.ptr));
    return .{ .cu_module = module };
}

pub fn load_ptx_text(
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

pub fn get_func(self: *const Device, module_name: []const u8, func_name: []const u8) !?CudaFunction {
    const modules = self.modules orelse return null;
    var h = modules.get(module_name) orelse return null;
    return .{ .cu_function = h.get_func(func_name) orelse return null };
}

pub fn default() CudaError.Error!Device {
    return Device.new(0);
}

pub fn alloc(self: Device, comptime T: type, length: usize) CudaError.Error!CudaSlice(T) {
    var device_ptr: cuda.CUdeviceptr = undefined;
    try Error.fromCudaErrorCode(cuda.cuMemAlloc(&device_ptr, @sizeOf(T) * length));
    return .{ .device_ptr = device_ptr, .len = length, .device = self };
}
pub fn alloc_zeros(self: Device, comptime data_type: type, length: usize) CudaError.Error!CudaSlice(data_type) {
    const slice = try self.alloc(data_type, length);
    try Device.memset_zeros(data_type, slice);
    return slice;
}
pub fn memset_zeros(comptime data_type: type, cuda_slice: CudaSlice(data_type)) CudaError.Error!void {
    try Error.fromCudaErrorCode(cuda.cuMemsetD8_v2(cuda_slice.device_ptr, 0, cuda_slice.len * @sizeOf(data_type)));
}
pub fn free(self: *const Device) void {
    Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRelease(self.device)) catch |err| @panic(@errorName(err));
}
pub fn htod_copy_into(comptime T: type, src: []const T, destination: CudaSlice(T)) CudaError.Error!void {
    try Error.fromCudaErrorCode(cuda.cuMemcpyHtoD_v2(destination.device_ptr, @ptrCast(src), @sizeOf(T) * src.len));
}
pub fn htod_copy(self: Device, comptime T: type, src: []const T) CudaError.Error!CudaSlice(T) {
    const slice = try self.alloc(T, src.len);
    try Device.htod_copy_into(T, src, slice);
    return slice;
}
pub fn sync_reclaim(comptime T: type, allocator: std.mem.Allocator, slice: CudaSlice(T)) !std.ArrayList(T) {
    var h_Array = try std.ArrayList(T).initCapacity(allocator, slice.len);
    try Error.fromCudaErrorCode(cuda.cuMemcpyDtoH_v2(@ptrCast(h_Array.items), slice.device_ptr, @sizeOf(T) * slice.len));
    // Zig doesn't know the item pointer length, so setting the length of slice.
    h_Array.items.len = slice.len;
    return h_Array;
}
