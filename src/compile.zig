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

    fn to_params(self: @This(), allocator: std.mem.Allocator) CompileError!std.ArrayList([]const u8) {
        var arr = std.ArrayList([]const u8){};
        // helper variables
        var tmp: []u8 = undefined;
        var sentinel: []u8 = undefined;

        if (self.ftz) |field| {
            tmp = try std.fmt.allocPrint(allocator, "--ftz={any}", .{field});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        if (self.prec_sqrt) |field| {
            tmp = try std.fmt.allocPrint(allocator, "--prec-sqrt={any}", .{field});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        if (self.prec_div) |field| {
            tmp = try std.fmt.allocPrint(allocator, "--prec-div={any}", .{field});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        if (self.use_fast_math) |field| {
            tmp = try std.fmt.allocPrint(allocator, "--fmad={any}", .{field});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        if (self.maxrregcount) |field| {
            tmp = try std.fmt.allocPrint(allocator, "--maxrregcount={d}", .{field});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        for (self.arch) |arch| {
            tmp = try std.fmt.allocPrint(allocator, "-arch={s}", .{arch});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        for (self.macro) |macro| {
            tmp = try std.fmt.allocPrint(allocator, "--define-macro={s}", .{macro});
            const param = try allocator.alloc(u8, tmp.len + 1);
            std.mem.copyForwards(u8, param[0..tmp.len], tmp);
            param[tmp.len] = 0;
            try arr.append(allocator, @as([]const u8, param[0 .. tmp.len + 1]));
            allocator.free(tmp);
        }
        for (self.include_paths) |path| {
            tmp = try std.fmt.allocPrint(allocator, "--include-path={s}", .{path});
            sentinel = try allocator.allocSentinel(u8, tmp.len, 0);
            std.mem.copyForwards(u8, sentinel[0..tmp.len], tmp);
            try arr.append(allocator, @as([]const u8, sentinel));
            allocator.free(tmp);
            allocator.free(sentinel);
        }
        return arr;
    }
};
pub fn cudaText(cuda_text: []const u8, options: ?Options, allocator: std.mem.Allocator) CompileError![:0]const u8 {
    var program: nvrtc.nvrtcProgram = undefined;
    try Error.fromNvrtcErrorCode(nvrtc.nvrtcCreateProgram(&program, cuda_text.ptr, null, 0, null, null));
    try cudaProgram(program, options, allocator);
    const ptx_data = try getPtx(program, allocator);
    return ptx_data;
}
pub fn cudaFile(cuda_path: std.fs.File, options: ?Options, allocator: std.mem.Allocator) ![:0]const u8 {
    const data = try cuda_path.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(data);

    const data_with_zero = try allocator.alloc(u8, data.len + 1);
    std.mem.copyForwards(u8, data_with_zero[0..data.len], data);
    data_with_zero[data.len] = 0;
    defer allocator.free(data_with_zero);

    return try cudaText(data_with_zero[0 .. data.len + 1], options, allocator);
}

pub fn cudaProgram(prg: nvrtc.nvrtcProgram, options: ?Options, allocator: std.mem.Allocator) CompileError!void {
    const n_o = if (options != null) options.? else @as(Options, .{});
    var parmas = try n_o.to_params(allocator);
    const params_own = try parmas.toOwnedSlice(allocator);
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
    return try ptx.toOwnedSliceSentinel(allocator, 0);
}
