const std = @import("std");
const Translator = @import("translate_c").Translator;

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // exe points to main.zig that uses cudaz
    const exe = b.addExecutable(.{ .name = "main", .root_module = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize }) });

    // Translate the custom C type into a Zig module
    const translate_c_dep = b.dependency("translate_c", .{ .target = target, .optimize = optimize });
    const tuple_translate: Translator = .init(translate_c_dep, .{
        .name = "tuple_c",
        .c_source_file = b.path("c/tuple.h"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("tuple_c", tuple_translate.mod);

    // Point to cudaz dependency
    const cudaz_dep = b.dependency(
        "cudaz",
        .{}, // replace with `.{ .CUDA_PATH = @as([]const u8, "<your cuda path>") }` to specify custom CUDA_PATH
    );

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.root_module.addImport("cudaz", cudaz_module);

    // Dynamically link to libc, cuda, nvrtc
    exe.root_module.link_libc = true;
    exe.root_module.linkSystemLibrary("cuda", .{});
    exe.root_module.linkSystemLibrary("nvrtc", .{});

    // Run binary
    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}
