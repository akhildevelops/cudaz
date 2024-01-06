const std = @import("std");
const TestFiles = struct {
    test_dir: std.fs.Dir,
    /// The caller owns the memory of the slice returned from this method.
    pub fn all(self: *const TestFiles, allocator: std.mem.Allocator) ![][]const u8 {
        var test_files = std.ArrayList([]const u8).init(allocator);
        defer test_files.deinit();
        var walker = try self.test_dir.walk(allocator);
        defer walker.deinit();
        while (try walker.next()) |item| {
            if (item.kind == .file) {
                try test_files.append(try allocator.dupe(u8, item.path));
            }
        }
        return test_files.toOwnedSlice();
    }
    pub fn next(self: *TestFiles) ?[]const u8 {
        _ = self; // autofix
    }
};
pub fn testFiles(x: []const u8) !TestFiles {
    const test_dir = try std.fs.cwd().openDir(x, .{ .iterate = true });
    return .{ .test_dir = test_dir };
}

test "files" {
    const files = try testFiles("test");
    const files_a = try files.all(std.testing.allocator);
    for (files_a) |file| {
        std.testing.allocator.free(file);
    }
    std.testing.allocator.free(files_a);
}
