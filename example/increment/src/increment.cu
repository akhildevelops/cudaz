extern "C" __global__ void increment(float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = out[i] + 1;
}