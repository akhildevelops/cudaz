pub const PathBuffer = struct {
    raw_path: []const u8,
    const Self = @This();

    fn new(path: []const u8) Self {
        return .{ .raw_path = path };
    }
};
