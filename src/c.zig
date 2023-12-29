pub const cuda = @cImport({
    @cInclude("cuda.h");
    // @cInclude("cuda_runtime.h");
    // @cInclude("curand.h");
});

pub const curand = @cImport(@cInclude("curand.h"));

pub const nvrtc = @cImport(@cInclude("nvrtc.h"));
