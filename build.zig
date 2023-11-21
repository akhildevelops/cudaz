// Build file to create executables and small util binaries like clean to remove cached-dirs and default artifact folder.
const std = @import("std");

pub fn build(b: *std.Build) void {
    ////////////////////////////////////////////////////////////
    //// Creates default options for building the library.
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const CUDA_HEADER = "/usr/local/cuda/targets/x86_64-linux/include";

    ////////////////////////////////////////////////////////////
    //// Static Library
    // Creates a compile step to create static cudaz library
    const lib = b.addStaticLibrary(.{
        .name = "cudaz",
        .root_source_file = .{ .path = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Links libc for dynamic loading
    lib.linkLibC();

    // Path to cuda headers
    lib.addIncludePath(.{ .path = CUDA_HEADER });

    // Links libcuda.so as dynamic lib
    lib.linkSystemLibrary("cuda");

    // Installs the artifact into zig-out dir
    b.installArtifact(lib);

    ////////////////////////////////////////////////////////////
    //// Unit Testing
    // Creates a test binary.
    const main_tests = b.addTest(.{ .root_source_file = .{ .path = "./test.zig" }, .target = target, .optimize = optimize });

    // Link libc
    main_tests.linkLibC();

    // Path to cuda headers
    main_tests.addIncludePath(.{ .path = CUDA_HEADER });

    // Links libcuda.so as dynamic lib
    main_tests.linkSystemLibrary("cuda");

    // Creates a run step for test binary
    const run_main_tests = b.addRunArtifact(main_tests);

    // Test step is created to be run from commandline i.e, zig build test
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);

    ////////////////////////////////////////////////////////////
    //// Clean the cache folders and artifacts

    // Creates a binary that cleans up zig artifact folders
    const clean = b.addExecutable(.{ .name = "clean", .root_source_file = .{ .path = "bin/delete-zig-cache.zig" }, .target = target, .optimize = optimize });

    // Creates a run step
    const clean_step = b.addRunArtifact(clean);

    // Register clean command i.e, zig build clean to cleanup any artifacts and cache
    const clean_cmd = b.step("clean", "Cleans the cache folders");
    clean_cmd.dependOn(&clean_step.step);
}
