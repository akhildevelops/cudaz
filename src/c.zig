pub const cuda = @cImport({
    @cInclude("cuda.h");
});

pub const nvrtc = @cImport(@cInclude("nvrtc.h"));
