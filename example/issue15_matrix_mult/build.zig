const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // `-Drepro=true` runs the launch call exactly like issue #15 does
    // (scalar kernel arguments passed by value). It is expected to fail with
    // an invalid launch (SIGSEGV on the reporter's driver).
    const repro = b.option(
        bool,
        "repro",
        "Reproduce issue #15's segfault (scalar kernel args passed by value)",
    ) orelse false;

    // `-Dzig_ptx=true` loads PTX generated from the original Zig kernel by
    // Zig's NVPTX backend instead of compiling matmul.cu with NVRTC.
    const zig_ptx = b.option(
        bool,
        "zig_ptx",
        "Compile the original Zig kernel to PTX with Zig's LLVM NVPTX backend",
    ) orelse false;

    const options = b.addOptions();
    options.addOption(bool, "repro", repro);
    options.addOption(bool, "zig_ptx", zig_ptx);

    const exe = b.addExecutable(.{
        .name = "issue15-matrix-mult",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.root_module.addImport("build_options", options.createModule());

    // Zig can emit PTX through LLVM's NVPTX backend, but it cannot emit an
    // nvptx64 binary. `-fno-emit-bin` + `-femit-asm` is what makes it work.
    const ptx_cmd = b.addSystemCommand(&.{ b.graph.zig_exe, "build-obj" });
    ptx_cmd.addFileArg(b.path("src/MMmult.zig"));
    ptx_cmd.addArgs(&.{
        "-target",       "nvptx64-freestanding",
        "-O",            "ReleaseFast",
        "-fno-emit-bin",
    });
    const ptx_file = ptx_cmd.addOutputFileArg2("MMmult.ptx", .{ .prefix = "-femit-asm=" });
    exe.root_module.addAnonymousImport("MMmult_ptx", .{ .root_source_file = ptx_file });

    const cudaz_dep = b.dependency("cudaz", .{});
    exe.root_module.addImport("cudaz", cudaz_dep.module("cudaz"));

    exe.root_module.link_libc = true;
    exe.root_module.linkSystemLibrary("cuda", .{});
    exe.root_module.linkSystemLibrary("nvrtc", .{});

    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}
