const std = @import("std");

pub fn build(b: *std.Build) !void {
    // exe points to main.zig that uses cudaz
    const exe = b.addExecutable(.{ .name = "main", .root_module = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = b.standardTargetOptions(.{}) }) });

    // Point to cudaz dependency
    const cudaz_dep = b.dependency(
        "cudaz",
        .{}, // replace with `.{ .CUDA_PATH = @as([]const u8, "<your cuda path>") }` to specify custom CUDA_PATH
    );

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.root_module.addImport("cudaz", cudaz_module);

    // Dynamically link to libc, cuda, nvrtc
    exe.linkLibC();
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("nvrtc");

    // Run binary
    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}
