pub const cuda = @cImport({
    @cInclude("cuda.h");
});

pub const curand = @cImport(@cInclude("curand.h"));

pub const nvrtc = @cImport(@cInclude("nvrtc.h"));
