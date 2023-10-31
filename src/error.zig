// const Error = error{generating_cuda_error};
const std = @import("std");
const Type = std.builtin.Type;
const cuda = @import("cuda.zig");
const utils = @import("utils.zig");
fn CudaErrorsToEnum(comptime tag_type: type) type {
    switch (@typeInfo(cuda)) {
        .Struct => |x| {
            const null_decls: []const Type.Declaration = &.{};
            var errors: [100]Type.EnumField = undefined;
            var counter: usize = 0;
            @setEvalBranchQuota(100000);
            for (x.decls) |declaration| {
                if (std.mem.startsWith(u8, declaration.name, "CUDA_ERROR")) {
                    errors[counter] = .{ .name = declaration.name, .value = @field(cuda, declaration.name) };
                    counter += 1;
                }
            }

            return @Type(Type{ .Enum = .{ .tag_type = tag_type, .fields = errors[0..counter], .decls = null_decls, .is_exhaustive = false } });
        },
        else => @compileError("Cannot generate error type"),
    }
}
// const Self = @This();
pub fn fromCudaErrorCode(error_code: u32) !void {
    if (error_code == 0) return;
    std.debug.print("{d}", .{error_code});
    const cuda_error_enums = comptime CudaErrorsToEnum(u32);
    const some_struct = utils.EnumToError(cuda_error_enums);
    const val = some_struct.from_error_code(error_code).?;
    std.debug.print("{any}", .{val});
    return val;
}

test "Test_Gen_Errors" {
    //zig test error.zig -lcuda -I /usr/local/cuda-12.2/targets/x86_64-linux/include
    const cuda_error_enums = comptime CudaErrorsToEnum(u32);
    const some_struct = utils.EnumToError(cuda_error_enums);
    try std.testing.expect(some_struct.from_error_code(1) orelse unreachable == error.CUDA_ERROR_INVALID_VALUE);
}
