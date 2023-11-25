const std = @import("std");
const path = @import("path.zig");
const testing = std.testing;
const cuda = @import("cuda.zig");
const Error = @import("error.zig");
const CudaError = Error.CudaError;

pub const Device = struct {
    device: cuda.CUdevice,
    primary_context: cuda.CUcontext,
    ordinal: u16,
    const Self = @This();

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

    pub fn get_func(self: *const Self, module_name: []const u8, func_name: []const u8) !?CudaFunction {
        const modules = self.modules orelse return null;
        var h = modules.get(module_name) orelse return null;
        return .{ .cu_function = h.get_func(func_name) orelse return null };
    }

    pub fn default() CudaError.Error!Self {
        return Self.new(0);
    }

    pub fn alloc(self: Self, comptime T: type, length: usize) CudaError.Error!CudaSlice(T) {
        var device_ptr: cuda.CUdeviceptr = undefined;
        try Error.fromCudaErrorCode(cuda.cuMemAlloc(&device_ptr, @sizeOf(T) * length));
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
    pub fn free(self: *const Self) void {
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

const CudaFunction = struct { cu_function: cuda.CUfunction };

fn CudaSlice(comptime T: type) type {
    return struct {
        device_ptr: cuda.CUdeviceptr,
        len: usize,
        device: Device,
        pub const element_type: type = T;
        pub fn clone(self: @This()) CudaError.Error!@This() {
            const c_slice = try self.device.alloc(T, self.len);
            try Error.fromCudaErrorCode(cuda.cuMemcpyDtoD_v2(c_slice.device_ptr, self.device_ptr, @sizeOf(T) * self.len));
            return c_slice;
        }
        pub fn free(self: @This()) void {
            Error.fromCudaErrorCode(cuda.cuMemFree_v2(self.device_ptr)) catch |err| @panic(@errorName(err));
        }
    };
}

pub const LaunchConfig = struct {
    grid_dim: struct { u32, u32, u32 },
    block_dim: struct { u32, u32, u32 },
    shared_mem_bytes: u32,
    const Self = @This();
    pub fn for_num_elems(n: u32) Self {
        const NUM_THREADS: u32 = 1024;
        const num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        return .{
            .grid_dim = .{ num_blocks, 1, 1 },
            .block_dim = .{ NUM_THREADS, 1, 1 },
            .shared_mem_bytes = 0,
        };
    }
};

pub const Function = struct {
    cu_func: cuda.CUfunction,
    pub fn run(self: @This(), params: anytype, cfg: LaunchConfig) CudaError.Error!void {
        if (@typeInfo(@TypeOf(params)) != .Struct) return error.params_not_struct;
        var e_params: ?*u32 = null;
        try Error.fromCudaErrorCode(cuda.cuLaunchKernel(self.cu_func, cfg.grid_dim[0], cfg.grid_dim[1], cfg.grid_dim[2], cfg.block_dim[0], cfg.block_dim[1], cfg.block_dim[2], cfg.shared_mem_bytes, null, @ptrCast(@constCast(&params)), @ptrCast(&e_params)));
    }
};

pub const Module = struct {
    cu_module: cuda.CUmodule,

    pub fn get_func(self: @This(), name: []const u8) CudaError.Error!Function {
        var function: cuda.CUfunction = undefined;
        try Error.fromCudaErrorCode(cuda.cuModuleGetFunction(&function, self.cu_module, name.ptr));
        return .{ .cu_func = function };
    }
};
