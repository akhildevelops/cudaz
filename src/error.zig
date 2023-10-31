// const Error = error{generating_cuda_error};
const std = @import("std");
const Type = std.builtin.Type;
const cuda = @import("cuda.zig");
const utils = @import("utils.zig");
fn gen_cuda_error() type {
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

            return @Type(Type{ .Enum = .{ .tag_type = u16, .fields = errors[0..counter], .decls = null_decls, .is_exhaustive = false } });
        },
        else => @compileError("Cannot generate error type"),
    }
}

test "Test_Gen_Errors" {
    const Error = comptime gen_cuda_error();
    // switch (@typeInfo(Error)) {
    //     .Enum => |error_elements| {
    //         for (error_elements orelse unreachable) |e| {
    //             std.debug.print("{s}\n", .{e.name});
    //         }
    //     },
    //     else => @compileError("Test failed"),
    // }
    // std.debug.print("{any}", .{@typeInfo(Error)});
    const some_struct = utils.EnumToError(Error);
    std.debug.print("{any}\n\n", .{some_struct.from_error_code(1)});
    // @compileLog("Reached");
}
