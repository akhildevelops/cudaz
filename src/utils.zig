const std = @import("std");
const Type = std.builtin.Type;
pub fn EnumToError(comptime some_type: type) type {
    switch (@typeInfo(some_type)) {
        .@"enum" => |enum_type| {
            var error_names: [enum_type.fields.len]Type.Error = undefined;
            for (enum_type.fields, 0..) |field, index| {
                error_names[index] = .{ .name = field.name };
            }
            const error_type = @Type(std.builtin.Type{ .error_set = &error_names });
            return struct {
                pub const Error: type = error_type;
                pub fn from_error_code(x: enum_type.tag_type) ?error_type {
                    const enum_val: some_type = @enumFromInt(x);
                    inline for (@typeInfo(error_type).error_set orelse unreachable) |error_val| {
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

pub const DType = enum(u8) {
    f16 = 0,
    f32 = 1,
    pub fn size(self: DType) usize {
        return switch (self) {
            .f16 => @sizeOf(f16),
            .f32 => @sizeOf(f32),
        };
    }
};

test "enum_to_error" {
    const SomeEnum = enum(u16) { one = 1, thousand = 1000 };
    const error_type = EnumToError(SomeEnum);
    try std.testing.expect(error_type.from_error_code(1000) orelse unreachable == error.thousand);
    try std.testing.expect(error_type.from_error_code(1) orelse unreachable == error.one);
}
