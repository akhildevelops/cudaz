const std = @import("std");
const path = @import("path.zig");
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
    modules: ?std.StringHashMap(CudaModule),
    const Self = @This();

    pub fn load_ptx(self: *Self, file_path: path.PathBuffer, module_name: []const u8, function_names: [][]const u8, allocator: std.mem.Allocator) !void {
        var module: cuda.CUmodule = undefined;
        try Error.fromCudaErrorCode(cuda.cuModuleLoad(&module, file_path.raw_path.ptr));
        var mapping = std.StringHashMap(cuda.CUfunction).init(allocator);
        for (function_names) |name| {
            var function: cuda.CUfunction = undefined;
            try Error.fromCudaErrorCode(cuda.cuModuleGetFunction(&function, module, name.ptr));
            try mapping.put(name, function);
        }
        const lmodule: CudaModule = .{ .name = module_name, .functions = mapping };
        self.modules = std.StringHashMap(CudaModule).init(allocator);
        try self.modules.?.put(module_name, lmodule);
    }

    pub fn new(gpu: u16) CudaError.Error!Self {
        var device: cuda.CUdevice = undefined;
        var context: cuda.CUcontext = undefined;
        try Error.fromCudaErrorCode(cuda.cuInit(0));
        try Error.fromCudaErrorCode(cuda.cuDeviceGet(&device, gpu));
        try Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRetain(&context, device));
        try Error.fromCudaErrorCode(cuda.cuCtxSetCurrent(context));
        return .{ .device = device, .primary_context = context, .ordinal = gpu, .modules = null };
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
    pub fn free(self: *Self) void {
        if (self.modules) |*modules| {
            var module_iter = modules.valueIterator();
            for (0..@as(usize, modules.count())) |index| {
                module_iter.items[index].deinit();
            }
            modules.deinit();
        }
        Error.fromCudaErrorCode(cuda.cuDevicePrimaryCtxRelease(self.device)) catch |err| @panic(@errorName(err));
    }
    pub fn htod_copy_into(self: Self, comptime T: type, src: []const T, destination: CudaSlice(T)) CudaError.Error!void {
        _ = self;
        try Error.fromCudaErrorCode(cuda.cuMemcpyHtoD_v2(destination.device_ptr, @ptrCast(src), @sizeOf(T) * src.len));
    }
    pub fn launch_func(self: Self, func: CudaFunction, cfg: LaunchConfig, params: anytype) !void {
        _ = self;
        if (@typeInfo(@TypeOf(params)) != .Struct) return error.params_not_struct;
        inline for (@typeInfo(@TypeOf(params)).Struct.fields) |field| {
            std.debug.print("{s}:{d}:{any}:{d}\n\n", .{ field.name, field.alignment, @typeInfo(field.type), @sizeOf(field.type) });
        }
        std.debug.print("{d}\n", .{@sizeOf(@TypeOf(params))});
        // var k_params: ?*anytype = null;
        var e_params: ?*u32 = null;
        try Error.fromCudaErrorCode(cuda.cuLaunchKernel(func.cu_function, cfg.grid_dim[0], cfg.grid_dim[1], cfg.grid_dim[2], cfg.block_dim[0], cfg.block_dim[1], cfg.block_dim[2], cfg.shared_mem_bytes, null, @ptrCast(@constCast(&params)), @ptrCast(&e_params)));
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
        device: CudaDevice,
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

const CudaModule = struct {
    name: []const u8,
    functions: std.StringHashMap(cuda.CUfunction),
    const Self = @This();
    pub fn deinit(self: *Self) void {
        self.functions.deinit();
    }
    pub fn get_func(self: *const Self, func_name: []const u8) ?cuda.CUfunction {
        return self.functions.get(func_name);
    }
};
pub const LaunchConfig = struct {
    grid_dim: struct { u32, u32, u32 },
    block_dim: struct { u32, u32, u32 },
    shared_mem_bytes: u32,
    const Self = @This();
    pub fn for_num_elems(n: u32) Self {
        const NUM_THREADS: u32 = 1024;
        var num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        return .{
            .grid_dim = .{ num_blocks, 1, 1 },
            .block_dim = .{ NUM_THREADS, 1, 1 },
            .shared_mem_bytes = 0,
        };
    }
};
