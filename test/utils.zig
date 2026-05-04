const std = @import("std");
const TestFiles = struct {
    test_dir: std.Io.Dir,
    /// The caller owns the memory of the slice returned from this method.
    pub fn all(self: *const TestFiles, allocator: std.mem.Allocator, io: std.Io) ![][]const u8 {
        var test_files = std.ArrayList([]const u8).empty;
        defer test_files.deinit(allocator);
        var walker = try self.test_dir.walk(allocator);
        defer walker.deinit();
        while (try walker.next(io)) |item| {
            if (item.kind == .file) {
                try test_files.append(allocator, try allocator.dupe(u8, item.path));
            }
        }
        return test_files.toOwnedSlice(allocator);
    }
    pub fn next(self: *TestFiles) ?[]const u8 {
        _ = self; // autofix
    }
};
pub fn testFiles(x: []const u8, io: std.Io) !TestFiles {
    const test_dir = try std.Io.Dir.cwd().openDir(io, x, .{ .iterate = true });
    return .{ .test_dir = test_dir };
}

test "files" {
    const files = try testFiles("test", std.testing.io);
    defer files.test_dir.close(std.testing.io);
    const files_a = try files.all(std.testing.allocator, std.testing.io);
    for (files_a) |file| {
        std.testing.allocator.free(file);
    }
    std.testing.allocator.free(files_a);
}
