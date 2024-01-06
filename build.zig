// Build file to create executables and small util binaries like clean to remove cached-dirs and default artifact folder.
const std = @import("std");
const utils = @import("test/utils.zig");
fn getCudaPath(path: ?[]const u8, allocator: std.mem.Allocator) ![]const u8 {

    // Return of cuda_parent folder confirms presence of include directory.
    return incldue_path: {
        if (path) |parent| {
            const cuda_file = try std.fmt.allocPrint(allocator, "{s}/include/cuda.h", .{parent});
            defer allocator.free(cuda_file);
            _ = std.fs.openDirAbsolute(cuda_file, .{}) catch |e| {
                switch (e) {
                    std.fs.File.OpenError.FileNotFound => {
                        return error.CUDA_INSTALLATION_NOT_FOUND;
                    },
                    else => return e,
                }
            };
            // Ignore /cuda.h
            break :incldue_path parent;
        } else {
            const probable_roots = [_][]const u8{
                "/usr",
                "/usr/local/cuda",
                "/opt/cuda",
                "/usr/lib/cuda",
            };
            inline for (probable_roots) |parent| h: {
                const cuda_file = parent ++ "/include/cuda.h";
                _ = std.fs.openFileAbsolute(cuda_file, .{}) catch {
                    break :h;
                };
                // Ignore /cuda.h
                break :incldue_path parent;
            }
            return error.CUDA_INSTALLATION_NOT_FOUND;
        }
    };
}

pub fn build(b: *std.Build) !void {
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

    // Create cudaz module
    const cudaz_module = b.createModule(.{ .source_file = .{ .path = "src/lib.zig" } });
    try b.modules.put("cudaz", cudaz_module);

    const cuda_path = b.option([]const u8, "CUDA_PATH", "locally installed Cuda's path");

    const cuda_folder = try getCudaPath(cuda_path, b.allocator);

    const cuda_include_dir = try std.fmt.allocPrint(b.allocator, "{s}/include", .{cuda_folder});

    // const HEADER_FILES = std.ComptimeStringMap([]const u8, .{.{ "cuda", "/usr/local/cuda/targets/x86_64-linux/include" }});

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
    lib.addIncludePath(.{ .path = cuda_include_dir });

    const lib_paths = [_][]const u8{
        "lib",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    };
    inline for (lib_paths) |lib_path| {
        const path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ cuda_folder, lib_path });
        lib.addLibraryPath(.{ .path = path });
    }

    // Link cuda and nvrtc libraries
    lib.linkSystemLibrary("cuda");
    lib.linkSystemLibrary("nvrtc");

    // Installs the artifact into zig-out dir
    b.installArtifact(lib);

    ////////////////////////////////////////////////////////////
    //// Unit Testing
    // Creates a test binary.
    // Test step is created to be run from commandline i.e, zig build test
    const test_step = b.step("test", "Run library tests");

    const test_dir = try std.fs.cwd().openDir("test", .{ .iterate = true });
    var dir_iterator = try test_dir.walk(b.allocator);
    while (try dir_iterator.next()) |item| {
        if (item.kind == .file) {
            const test_path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ "test", item.path });
            const sub_test = b.addTest(.{ .root_source_file = .{ .path = test_path }, .target = target, .optimize = optimize });
            // Add Module
            sub_test.addModule("cudaz", cudaz_module);
            // Link libc
            sub_test.linkLibC();

            // Path to cuda headers
            sub_test.addIncludePath(.{ .path = cuda_include_dir });
            inline for (lib_paths) |lib_path| {
                const path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ cuda_folder, lib_path });
                sub_test.addLibraryPath(.{ .path = path });
            }
            // Link cuda and nvrtc libraries
            sub_test.linkSystemLibrary("cuda");
            sub_test.linkSystemLibrary("nvrtc");

            // Creates a run step for test binary
            const run_sub_tests = b.addRunArtifact(sub_test);
            test_step.dependOn(&run_sub_tests.step);
        }
    }

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
