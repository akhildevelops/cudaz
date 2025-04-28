const cuda = @import("c.zig").cuda;
const LaunchConfig = @import("launchconfig.zig").LaunchConfig;
const Device = @import("device.zig");
const Error = @import("error.zig");
const CudaError = Error.CudaError;
const CudaFunction = struct { cu_function: cuda.CUfunction };
const DType = @import("utils.zig").DType;
pub fn CudaSlice(comptime T: type) type {
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
pub const CudaSliceR = struct {
    device_ptr: cuda.CUdeviceptr,
    len: usize,
    device: Device,
    element_type: DType,
    pub fn clone(self: *const CudaSliceR) CudaError.Error!CudaSliceR {
        const c_slice = try self.device.allocR(self.element_type, self.len);
        try Error.fromCudaErrorCode(cuda.cuMemcpyDtoD_v2(c_slice.device_ptr, self.device_ptr, self.element_type.size() * self.len));
        return c_slice;
    }
    pub fn free(self: CudaSliceR) void {
        Error.fromCudaErrorCode(cuda.cuMemFree_v2(self.device_ptr)) catch |err| @panic(@errorName(err));
    }
};

pub const Function = struct {
    cu_func: cuda.CUfunction,
    pub fn run(self: @This(), params: anytype, cfg: LaunchConfig) CudaError.Error!void {
        if (@typeInfo(@TypeOf(params)) != .@"struct") @compileError("Invalid params type, must be a struct");
        var e_params: ?*u32 = null;
        try Error.fromCudaErrorCode(cuda.cuLaunchKernel(self.cu_func, cfg.grid_dim[0], cfg.grid_dim[1], cfg.grid_dim[2], cfg.block_dim[0], cfg.block_dim[1], cfg.block_dim[2], cfg.shared_mem_bytes, null, @ptrCast(@constCast(&params)), @ptrCast(&e_params)));
    }
};

pub const Module = struct {
    cu_module: cuda.CUmodule,

    pub fn getFunc(self: @This(), name: []const u8) CudaError.Error!Function {
        var function: cuda.CUfunction = undefined;
        try Error.fromCudaErrorCode(cuda.cuModuleGetFunction(&function, self.cu_module, name.ptr));
        return .{ .cu_func = function };
    }
};
