const std = @import("std");
fn foo() ?void {
    const x: ?u32 = null;
    if (x == null) return null;
    std.debug.print("{d}", .{x.?});
}

const y: struct { u32, u32, u32 } = .{ 1, 2, 3 };

pub fn main() !void {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Struct without fields can be defined as const y: struct { u32, u32, u32 } = .{ 1, 2, 3 }; that become a tuple
    // std.debug.print("{s}", .{@typeName(@TypeOf(y))});
    //y[0] = 5;
    //std.debug.print("{d}", .{y[0]});

    // _ = foo();
    // var z = &[_]u32{ 1, 2 };
    // std.debug.print("{s}", .{@typeName(@TypeOf(z))});
    // var x: [*]align(1) u32 = undefined;
    // std.debug.print("x:{*}\n", .{x});
    // std.debug.print("{d}\n", .{@typeInfo(@TypeOf(x)).Pointer.alignment});
    // var y: [*]u32 = @alignCast(x);
    // std.debug.print("{d}\n", .{@typeInfo(@TypeOf(x)).Pointer.alignment});
    // std.debug.print("x:{*},y:{*}\n", .{ x, y });
    // std.debug.print("{d}\n", .{@typeInfo(@TypeOf(y)).Pointer.alignment});
    // std.debug.print("x+1:{*},y+1:{*}\n", .{ x + 1, y + 1 });

    // var x: u64 align(1) = undefined;
    // // x[0] = 1;
    // // var y: [*]u32 = @ptrCast(&x);
    // // _ = y;
    // var v: *align(2) u32 = @ptrFromInt(0x7ffe3fd16d04); // if address is not divisible by alignment value, this throws error.
    // std.debug.print("v:{*}\n", .{v});

    // // If alignment can't be coerced from higher value to lower value then @aligncast shoulld be used.
    // // But aligncast will insert safety check for validating memory value to be multiple of alignment value of w i.e, 4
    // // If v's value is divisible by 2 but not divisible by 4 then zig panics on runtime giving alignment mismatch.
    // var w: [*]u32 = @alignCast(@ptrCast(v));
    // std.debug.print("w:{*}", .{w + 1});
    // var z: [*]align(2) u64 = @ptrCast(@alignCast(&x));
    // z[5] = 1;
    // std.debug.print("{any}\n,{any}\n", .{ @typeInfo(@TypeOf(&x)), @typeInfo(@TypeOf(z)) });
    // std.debug.print("{any}\n,{any}\n", .{ @typeInfo(@TypeOf(&z[0])), @typeInfo(@TypeOf(&z[6])) });
    // std.debug.print("&z[0]:{*},z+1:{*},&z[1]:{*},&z[4]:{*},&z[5]:{*},z[5]:{d},&z[6]:{*}\n\n", .{ &z[0], z + 1, &z[1], &z[4], &z[5], z[5], &z[6] });
    // std.debug.print("is divisible &z[0]:{d},z+1:{d},&z[1]:{d},&z[4]:{d},&z[5]:{d},&z[6]:{d}", .{ @intFromPtr(&z[0]) % 4, @intFromPtr(z + 1) % 4, @intFromPtr(&z[1]) % 4, @intFromPtr(&z[4]) % 4, @intFromPtr(&z[5]) % 4, @intFromPtr(&z[6]) % 4 });

    // General purpose allocator
    const GP = std.testing.allocator;
    var arr = std.ArrayList(u32).init(GP);
    _ = try arr.toOwnedSlice();
}

test "arraylist" {
    // const GP = std.testing.allocator;
    // var arr = std.ArrayList(u32).init(GP);
    // try arr.appendSlice(&[_]u32{ 1, 3, 0, 4, 5 });
    // const o = try arr.toOwnedSlice();
    // const m_o: [*]u32 = @ptrCast(o);
    // const slice = m_o[0..100];
    // std.debug.print("{any}:{any}:{d}\n", .{ @typeInfo(@TypeOf(&[_]u32{ 1, 2 })), &[_]u32{ 1, 2 }, slice[50] });
    // GP.free(o);
    var slice = &[_]u32{ 1, 2, 3 };
    slice.len = 100; //This fails with the error: cannot assign to constant (resonable)
    const m_slice: [*]u32 = @ptrCast(&slice);
    _ = m_slice[0..100][50];
}

test "bool_to_str" {
    // const t: bool = true;
    var arr = std.ArrayList(u8).init(std.testing.allocator);
    // defer arr.deinit();
    const writer = arr.writer();
    try std.fmt.format(writer, "{any}", .{true});
    const str = try arr.toOwnedSlice();
    std.debug.print("{s}\n", .{str});
    std.testing.allocator.free(str);
}

test "free_sentinel_slice" {
    var arr = std.ArrayList(u8).init(std.testing.allocator);
    const sentinel_slice = try arr.toOwnedSliceSentinel(0);
    std.testing.allocator.free(sentinel_slice.ptr);
}
