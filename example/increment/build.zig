const std = @import("std");

pub fn build(b: *std.Build) !void {
    // exe points to main.zig that uses cudaz
    const exe = b.addExecutable(.{ .name = "main", .root_source_file = .{ .path = "src/main.zig" } });

    // Add module, include cuda header files and libraries.

    // Point to cudaz dependency
    const cudaz_dep = b.dependency("cudaz", .{});

    // Fetch and add the module from cudaz dependency
    const cudaz_module = cudaz_dep.module("cudaz");
    exe.addModule("cudaz", cudaz_module);

    // Point to cudaz library and add obtained header files and library paths to exe.
    const cudaz_lib = cudaz_dep.artifact("cudaz");
    exe.addIncludePath(cudaz_lib.include_dirs.items[0].path);
    for (cudaz_lib.lib_paths.items) |path| {
        exe.addLibraryPath(path);
    }

    // Dynamically link to libc, cuda, nvrtc
    exe.linkLibC();
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("nvrtc");

    // Run binary
    const run = b.step("run", "Run the binary");
    const run_step = b.addRunArtifact(exe);
    run.dependOn(&run_step.step);
}
