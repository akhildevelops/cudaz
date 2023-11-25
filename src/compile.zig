const nvrtc = @cImport(@cInclude("nvrtc.h"));
const std = @import("std");
const PTX = struct { compiled_code: [][]const u8, allocator: std.mem.Allocator };
const Path = std.fs.path;
const Error = @import("error.zig");
const Options = struct {
    ftz: ?bool = null,
    prec_sqrt: ?bool = null,
    prec_div: ?bool = null,
    use_fast_math: ?bool = null,
    maxrregcount: ?usize = null,
    include_paths: [][]const u8 = &[_][]const u8{},
    arch: ?[][]const u8 = null,

    fn to_params(self: @This(), allocator: std.mem.Allocator) !std.ArrayList([:0]const u8) {
        var arr = std.ArrayList([:0]const u8).init(allocator);
        var ext_str: [:0]u8 = undefined;
        if (self.ftz) |field| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--ftz={any}", .{field});
            try arr.append(ext_str);
        }
        if (self.prec_sqrt) |field| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--prec-sqrt={any}", .{field});
            try arr.append(ext_str);
        }
        if (self.prec_div) |field| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--prec-div={any}", .{field});
            try arr.append(ext_str);
        }
        if (self.use_fast_math) |field| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--fmad={any}", .{field});
            try arr.append(ext_str);
        }
        if (self.maxrregcount) |field| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--maxrregcount={d}", .{field});
            try arr.append(ext_str);
        }
        for (self.include_paths) |path| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--include-path={s}", .{path});
            try arr.append(ext_str);
        }
        return arr;
    }
};

pub fn cudaFile(cuda_path: std.fs.File, options: ?Options, allocator: std.mem.Allocator) !void {
    var data = std.ArrayList(u8).init(allocator);
    try cuda_path.reader().readAllArrayList(&data, std.math.maxInt(usize));
    const data_sentinel = try data.toOwnedSliceSentinel(0);
    defer allocator.free(data_sentinel);
    var program: nvrtc.nvrtcProgram = undefined;

    try Error.fromNvrtcErrorCode(nvrtc.nvrtcCreateProgram(&program, data_sentinel.ptr, undefined, 0, undefined, undefined));
    try cudaProgram(program, options, allocator);
}

pub fn cudaProgram(prg: nvrtc.nvrtcProgram, options: ?Options, allocator: std.mem.Allocator) !void {
    const n_o = if (options != null) options.? else @as(Options, .{});
    var parmas = try n_o.to_params(allocator);
    const params_own = try parmas.toOwnedSlice();
    defer {
        for (params_own) |param| {
            allocator.free(param);
        }
        allocator.free(params_own);
    }
    const cparams = try allocator.alloc([*]const u8, params_own.len);
    defer allocator.free(cparams);
    for (params_own, cparams) |p, *cp| {
        cp.* = p.ptr;
    }
    try Error.fromNvrtcErrorCode(nvrtc.nvrtcCompileProgram(prg, std.math.cast(c_int, cparams.len).?, cparams.ptr));
}
