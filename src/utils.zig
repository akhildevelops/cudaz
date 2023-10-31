const std = @import("std");
const Type = std.builtin.Type;
pub fn EnumToError(comptime some_type: type) type {
    switch (@typeInfo(some_type)) {
        .Enum => |enum_type| {
            var error_names: [enum_type.fields.len]Type.Error = undefined;
            for (enum_type.fields, 0..) |field, index| {
                error_names[index] = .{ .name = field.name };
            }
            const error_type = @Type(std.builtin.Type{ .ErrorSet = &error_names });
            return struct {
                pub fn from_error_code(x: enum_type.tag_type) ?error_type {
                    const enum_val: some_type = @enumFromInt(x);
                    inline for (@typeInfo(error_type).ErrorSet orelse unreachable) |error_val| {
                        if (std.mem.eql(u8, error_val.name, @tagName(enum_val))) {
                            return @field(error_type, error_val.name);
                        }
                    }
                    return null;
                }
            };
        },
        else => @compileError("Cannot convert non enum type to Error type"),
    }
}

test "enum_to_error" {
    const Value = enum(u2) {
        zero = 0,
        one = 1,
        two = 2,
    };
    _ = Value;
    // const error_type = EnumToError(some_enum);
    // _ = error_type;
    // std.debug.print("{*}\n", .{@typeInfo(Value).Enum.decls.ptr});
    // std.debug.print("{any}\n", .{error_type.from_error_code(1000)});
}
