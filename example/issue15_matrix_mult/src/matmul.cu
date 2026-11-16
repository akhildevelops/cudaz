extern "C" __global__ void MMmult(
    unsigned int x,
    unsigned int y,
    unsigned int z,
    const float *A,
    const float *B,
    float *C)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= x * z)
    {
        return;
    }

    unsigned int row = gid / z;
    unsigned int col = gid % z;

    float sum = 0.0f;
    for (unsigned int i = 0; i < y; ++i)
    {
        sum += A[row * y + i] * B[i * z + col];
    }
    C[gid] = sum;
}
