# Examples
## [increment](increment)
Hello world project for cudaz that increments each value of an array using GPU.

## [Sending Custom Types to GPU](custom_type)
Send custom type to GPU apart from array of primitive types. This example sends an array of structs to GPU.

## [Issue #15 matrix multiplication](issue15_matrix_mult)
Reproduction and fix for the kernel-launch segfault reported in issue #15, with a CUDA equivalent of the original Zig SPIR-V kernel. Run `zig build run` for the fixed version, `zig build run -Dzig_ptx=true` to compile the original Zig kernel to PTX with Zig's NVPTX backend, or `zig build run -Drepro=true` to reproduce the invalid launch parameters.