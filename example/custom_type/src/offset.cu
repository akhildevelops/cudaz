// Below typedef can be replaced by #include<tuple.h> but currently cudaz's nvrtc cannot handle #include directives.
typedef struct
{
    float x;
    float y;
} tuple;

extern "C" __global__ void offset(tuple *in, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in[i].y - in[i].x;
}