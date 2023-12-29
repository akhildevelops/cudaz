const c = @import("c.zig");
const Error = @import("error.zig");
rng: c.curand.curandGenerator_t = undefined,
const DEFAULT_SEED = 0;
pub fn default() !@This() {
    var local_rng: c.curand.curandGenerator_t = undefined;
    try Error.fromCurandErrorCode(c.curand.curandCreateGenerator(&local_rng, c.curand.CURAND_RNG_PSEUDO_DEFAULT));
    try Error.fromCurandErrorCode(c.curand.curandSetPseudoRandomGeneratorSeed(local_rng, DEFAULT_SEED));
    return .{ .rng = local_rng };
}

pub fn init(seed: ?u64) !@This() {
    var local_rng: c.curand.curandGenerator_t = undefined;
    try Error.fromCurandErrorCode(c.curand.curandCreateGenerator(&local_rng, c.curand.CURAND_RNG_PSEUDO_DEFAULT));
    try Error.fromCurandErrorCode(c.curand.curandSetPseudoRandomGeneratorSeed(local_rng, seed orelse DEFAULT_SEED));
    return .{ .rng = local_rng };
}
