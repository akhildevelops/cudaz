const std = @import("std");
const nvrtc = @import("c.zig").nvrtc;
const PTX = struct { compiled_code: [][]const u8, allocator: std.mem.Allocator };
const Path = std.fs.path;
const Error = @import("error.zig");
const CompileError = std.mem.Allocator.Error || Error.NvrtcError.Error || error{StreamTooLong};

const Options = struct {
    ftz: ?bool = null,
    prec_sqrt: ?bool = null,
    prec_div: ?bool = null,
    use_fast_math: ?bool = null,
    maxrregcount: ?usize = null,
    include_paths: [][]const u8 = &[_][]const u8{},
    arch: [][]const u8 = &[_][]const u8{},
    macro: [][]const u8 = &[_][]const u8{},

    fn to_params(self: @This(), allocator: std.mem.Allocator) CompileError!std.ArrayList([:0]const u8) {
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
        for (self.arch) |arch| {
            ext_str = try std.fmt.allocPrintZ(allocator, "-arch={s}", .{arch});
            try arr.append(ext_str);
        }
        for (self.macro) |macro| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--define-macro={s}", .{macro});
            try arr.append(ext_str);
        }
        for (self.include_paths) |path| {
            ext_str = try std.fmt.allocPrintZ(allocator, "--include-path={s}", .{path});
            try arr.append(ext_str);
        }
        return arr;
    }
};
pub fn cudaText(cuda_text: [:0]const u8, options: ?Options, allocator: std.mem.Allocator) CompileError![:0]const u8 {
    var program: nvrtc.nvrtcProgram = undefined;
    try Error.fromNvrtcErrorCode(nvrtc.nvrtcCreateProgram(&program, cuda_text.ptr, null, 0, null, null));
    try cudaProgram(program, options, allocator);
    const ptx_data = try getPtx(program, allocator);
    return ptx_data;
}
pub fn cudaFile(cuda_path: std.fs.File, options: ?Options, allocator: std.mem.Allocator) ![:0]const u8 {
    var data = std.ArrayList(u8).init(allocator);
    try cuda_path.reader().readAllArrayList(&data, std.math.maxInt(usize));
    const data_sentinel = try data.toOwnedSliceSentinel(0);
    defer allocator.free(data_sentinel);
    return try cudaText(data_sentinel, options, allocator);
}

pub fn cudaProgram(prg: nvrtc.nvrtcProgram, options: ?Options, allocator: std.mem.Allocator) CompileError!void {
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
    Error.fromNvrtcErrorCode(nvrtc.nvrtcCompileProgram(prg, std.math.cast(c_int, cparams.len).?, cparams.ptr)) catch |e| {
        switch (e) {
            error.NVRTC_ERROR_COMPILATION => {
                var log_size: usize = undefined;
                try Error.fromNvrtcErrorCode(nvrtc.nvrtcGetProgramLogSize(prg, &log_size));
                const log_msg = try allocator.allocSentinel(u8, log_size, '0');
                try Error.fromNvrtcErrorCode(nvrtc.nvrtcGetProgramLog(prg, log_msg.ptr));
                std.debug.print("{s}\n", .{log_msg});
                allocator.free(log_msg);
            },
            else => return e,
        }
    };
    return {};
}

pub fn getPtx(prg: nvrtc.nvrtcProgram, allocator: std.mem.Allocator) CompileError![:0]const u8 {
    var ptx_size: usize = undefined;
    try Error.fromNvrtcErrorCode(nvrtc.nvrtcGetPTXSize(prg, &ptx_size));

    var ptx = try std.ArrayList(u8).initCapacity(allocator, ptx_size);
    ptx.expandToCapacity();
    try Error.fromNvrtcErrorCode(nvrtc.nvrtcGetPTX(prg, ptx.items.ptr));
    return try ptx.toOwnedSliceSentinel(0);
}
