// Original GPU code from https://github.com/akhildevelops/cudaz/issues/15
//
// This is a Zig SPIR-V kernel (`callconv(.kernel)` and the
// @workGroupId / @workGroupSize / @workItemId builtins). cudaz loads CUDA
// PTX/cubin through NVRTC + cuModuleLoadData, so this file is kept only as a
// reference and is not compiled by this example.
//
// It can be compiled to SPIR-V (not CUDA PTX) with:
//   zig build-obj src/MMmult.zig -target spirv64-vulkan
export fn MMmult(
    x: u32,
    y: u32,
    z: u32,
    A: [*]addrspace(.global) const f32,
    B: [*]addrspace(.global) const f32,
    C: [*]addrspace(.global) f32,
) callconv(.kernel) void {
    const gid = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const Cx = gid % x;
    const Cz = (gid - Cx) / z;
    C[gid] = 0;
    for (0..y) |i| {
        const Av = A + Cx + (i * y);
        const Bv = B + i + (Cz * z);
        const Cv = C + gid;
        Cv[0] += Av[0] * Bv[0];
    }
}
