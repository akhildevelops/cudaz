const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const cur_dir = std.Io.Dir.cwd();
    const zig_cache_dirs = [_][]const u8{ "zig-out", "zig-cache" };
    inline for (zig_cache_dirs) |dir| {
        try cur_dir.deleteTree(io, dir);
    }
}
