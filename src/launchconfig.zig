pub const LaunchConfig = struct {
    grid_dim: struct { u32, u32, u32 },
    block_dim: struct { u32, u32, u32 },
    shared_mem_bytes: u32,
    const Self = @This();
    pub fn for_num_elems(n: u32) Self {
        const NUM_THREADS: u32 = 1024;
        const num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        return .{
            .grid_dim = .{ num_blocks, 1, 1 },
            .block_dim = .{ NUM_THREADS, 1, 1 },
            .shared_mem_bytes = 0,
        };
    }
};
