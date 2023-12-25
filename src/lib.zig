pub const Device = @import("device.zig");
pub const Compile = @import("compile.zig");
pub const LaunchConfig = @import("launchconfig.zig").LaunchConfig;
pub usingnamespace @import("path.zig");

const Error = @import("error.zig");
pub const CudaError = Error.NvrtcError.Error || Error.NvrtcError.Error;
