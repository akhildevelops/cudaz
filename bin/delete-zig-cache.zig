const std = @import("std");
const fs = std.fs;
pub fn main() !void {
    const cur_dir = fs.cwd();
    const zig_cache_dirs = [_][]const u8{ "zig-out", "zig-cache" };
    inline for (zig_cache_dirs) |dir| {
        try cur_dir.deleteTree(dir);
    }
}
