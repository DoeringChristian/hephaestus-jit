/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#version 450 core
#pragma use_vulkan_memory_model
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_control_flow_attributes : enable

// M/N/K values filled out at pipeline creation time
#ifndef lM
#define lM 1
#endif

#ifndef lN
#define lN 1
#endif

#ifndef lK
#define lK 1
#endif

#ifndef TILE_M
#define TILE_M 1
#endif

#ifndef TILE_N
#define TILE_N 1
#endif

#ifndef TILE_K
#define TILE_K 1
#endif

const float alpha = 1.0;
const float beta = 1.0;

#ifndef B_COL_MAJOR
#define B_COL_MAJOR false
#endif

const uint A_ROW_LEN = TILE_K;
const uint A_NUM_ROWS = TILE_M;
const int A_LAYOUT = gl_CooperativeMatrixLayoutRowMajor;

#if B_COL_MAJOR
const uint B_ROW_LEN = TILE_K;
const uint B_NUM_ROWS = TILE_N;
const int B_LAYOUT = gl_CooperativeMatrixLayoutColumnMajor;
#else
const uint B_ROW_LEN = TILE_N;
const uint B_NUM_ROWS = TILE_K;
const int B_LAYOUT = gl_CooperativeMatrixLayoutRowMajor;
#endif

const int C_LAYOUT = gl_CooperativeMatrixLayoutRowMajor;
const int D_LAYOUT = gl_CooperativeMatrixLayoutRowMajor;

// layout(constant_id = 0) const uint lM = 1;
// layout(constant_id = 1) const uint lN = 1;
// layout(constant_id = 2) const uint lK = 1;
// layout(constant_id = 3) const uint TILE_M = 1;
// layout(constant_id = 4) const uint TILE_N = 1;
// layout(constant_id = 5) const uint TILE_K = 1;
// layout(constant_id = 6) const uint K = 1;
// layout(constant_id = 7) const uint strideA = 1;
// layout(constant_id = 8) const uint strideB = 1;
// layout(constant_id = 9) const uint strideC = 1;
// layout(constant_id = 10)const uint strideD = 1;
// layout(constant_id = 11)const float alpha = 1.0;
// layout(constant_id = 12)const float beta = 1.0;
// layout(constant_id = 13)const bool BColMajor = false;
// // Size and number of rows in A matrix. equal to TILE_K,TILE_M (always row major)
// layout(constant_id = 14)const uint A_ROW_LEN = 1;
// layout(constant_id = 15)const uint A_NUM_ROWS = 1;
// // Size and number of rows in B matrix. equal to TILE_N,TILE_K for row major and TILE_K,TILE_N for col major
// layout(constant_id = 16)const uint B_ROW_LEN = 1;
// layout(constant_id = 17)const uint B_NUM_ROWS = 1;

// #defines set on command line:
// A_BITS = 8 or 16 or 32 (bits per component)
// A_TYPE = e.g. float or float16_t
// C_BITS = 8 or 16 or 32 (bits per component)
// C_TYPE = e.g. float or float16_t
// coopmatT = fcoopmatNV, ucoopmatNV, scoopmatNV

// input bindings for A/B/C and the output
// bindings for A and B for uvec4/128-bit loads
// layout(buffer_reference) buffer InputAV4 { uvec4 x[]; } inputAV4;
// layout(buffer_reference) buffer InputBV4 { uvec4 x[]; } inputBV4;
// layout(buffer_reference) buffer InputC { C_TYPE x[]; } inputC;
// layout(buffer_reference) buffer Output { C_TYPE x[]; } outputO;


// NOTE: flip N and M here
layout(set = 0, binding = 0) buffer Config{
    uint32_t M;
    uint32_t N;
    uint32_t K;
}config_[5];
layout(set = 0, binding = 0) buffer InputAV4{
    uvec4 x[];
}inputAV4_[5];
layout(set = 0, binding = 0) buffer InputBV4{
    uvec4 x[];
}inputBV4_[5];
layout(set = 0, binding = 0) buffer InputC{
    C_TYPE x[];
}inputC_[5];
layout(set = 0, binding = 0) buffer OutputO{
    C_TYPE x[];
}outputO_[5];

#define config config_[0]
#define inputAV4 inputAV4_[1]
#define inputBV4 inputBV4_[2]
#define inputC inputC_[3]
#define outputO outputO_[4]

const int ELEMENTS_PER_VEC4 = 16/(A_BITS / 8); // 16 bytes, A_BITS bits per element
const int ROW_PAD_SH = ELEMENTS_PER_VEC4;

// Shared memory storage. Add a skew of ROW_PAD_SH bytes per row to avoid bank conflicts when accessing the shared memory
shared uvec4 Ash[A_NUM_ROWS * (A_ROW_LEN + ROW_PAD_SH) / ELEMENTS_PER_VEC4];
shared uvec4 Bsh[B_NUM_ROWS * (B_ROW_LEN + ROW_PAD_SH) / ELEMENTS_PER_VEC4];

const uint WORKGROUP_WIDTH_IN_SUBGROUPS = 4;
const uint WORKGROUP_HEIGHT_IN_SUBGROUPS = 2;
const uint NUM_SUBGROUPS = WORKGROUP_WIDTH_IN_SUBGROUPS * WORKGROUP_HEIGHT_IN_SUBGROUPS;
const uint INVOCATIONS_PER_WORKGROUP = SUBGROUP_SIZE * NUM_SUBGROUPS;
const uint C_ROWS = TILE_M / WORKGROUP_HEIGHT_IN_SUBGROUPS / lM;
const uint C_COLS = TILE_N / WORKGROUP_WIDTH_IN_SUBGROUPS / lN;
coopmat<C_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];

uint coordToOffset(uint i, uint j, uint stride, bool colMajor)
{
    return colMajor ? (stride * j + i) : (stride * i + j);
}

layout(local_size_x = INVOCATIONS_PER_WORKGROUP, local_size_y = 1, local_size_z = 1) in;
void main()
{
    // C is of size MxN

    uint M = config.M;
    uint N = config.N;
    uint K = config.K;

    uint strideA = K;
    uint strideB = B_COL_MAJOR ? K : N;
    uint strideC = N;
    uint strideD = N;
    
    // compute position in grid
    uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    uvec2 warpInTile = uvec2(gl_SubgroupID % WORKGROUP_WIDTH_IN_SUBGROUPS, gl_SubgroupID / WORKGROUP_WIDTH_IN_SUBGROUPS);

    // InputAV4 inputAV4 = params.inputAV4;
    // InputBV4 inputBV4 = params.inputBV4;
    // InputC inputC = params.inputC;
    // Output outputO = params.outputO;

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            result[i][j] = coopmat<C_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);
        }
    }

    uint chunkK = 0;

    // fetch A for the first iteration;
    const uint INVS_PER_ROW_A = A_ROW_LEN / ELEMENTS_PER_VEC4;
    uint atilek = ELEMENTS_PER_VEC4 * (gl_LocalInvocationID.x % INVS_PER_ROW_A);

    uvec4 temp_A[A_NUM_ROWS / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
    uint gabase = coordToOffset(TILE_M * tileID.y, chunkK, strideA, false);
    [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
        uint atilei = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
        temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)] = inputAV4.x[(gabase + strideA * atilei + atilek)/ELEMENTS_PER_VEC4];
    }

    // fetch B for the first iteration
    const uint INVS_PER_ROW_B = B_ROW_LEN / ELEMENTS_PER_VEC4;
    uint btilej = ELEMENTS_PER_VEC4 * (gl_LocalInvocationID.x % INVS_PER_ROW_B);

    uvec4 temp_B[B_NUM_ROWS / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];
    // gbbase is the anchor of this tile in global memory. It's computed from the
    // (k,j) coordinate based on whether the tile is column major. Within this tile,
    // the global->shared copy always works in terms of contiguous "rows" of memory.
    // So the addressing within a tile is not affected by B_COL_MAJOR.
    uint gbbase = coordToOffset(chunkK, TILE_N * tileID.x, strideB, B_COL_MAJOR);
    [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
        uint btilek = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
        temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)] = inputBV4.x[(gbbase + strideB * btilek + btilej)/ELEMENTS_PER_VEC4];
    }

    // Iterate over K.
    // On each iteration, the workgroup cooperates to memcpy a row of cooperative
    // matrices from matrix A into Ash and a column of cooperative matrices from
    // matrix B into Bsh. Then each subgroup loads the subset of those matrices
    // that it needs out of shared memory, and multiplies pairs of cooperative
    // matrices.
    for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {
        bool last = ((chunkK + TILE_K) >= K);

        const uint STRIDE_A_SH = (A_ROW_LEN + ROW_PAD_SH);

        // ensure that all threads in the subgroup finished reading from SMEM during the last iteration
        barrier();

        // store A from local storage to shared memory
        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint si = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            Ash[(STRIDE_A_SH * si + atilek) / ELEMENTS_PER_VEC4] = temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
        }

        const uint STRIDE_B_SH = (B_ROW_LEN + ROW_PAD_SH);

        // store B from local storage to shared memory
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint sk = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            Bsh[(STRIDE_B_SH * sk + btilej) / ELEMENTS_PER_VEC4] = temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];
        }

        // wait until all threads finished writing to shared memory before the math loop
        // Do this before fetching data for the next iteration so that the barrier does not
        // wait for the loads from global storage to be finished
        barrier();

        // we prefetch data from global memory as soon as possible to hide memory transfers
        // behind math
        // prefetch A
        uint gabase = coordToOffset(TILE_M * tileID.y, chunkK + TILE_K, strideA, false);
        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint atilei = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            if (!last) temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)] = inputAV4.x[(gabase + strideA * atilei + atilek) / ELEMENTS_PER_VEC4];
        }

        // prefetch B
        uint gbbase = coordToOffset(chunkK + TILE_K, TILE_N * tileID.x, strideB, B_COL_MAJOR);
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint btilek = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            if (!last) temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)] = inputBV4.x[(gbbase + strideB * btilek + btilej) / ELEMENTS_PER_VEC4];
        }

        // The actual math loop
        [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k)
        {
            uint sk = lK * k;

            // load A. A will be reused C_COLS times
            coopmat<A_TYPE, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                uint si = lM * (C_ROWS * warpInTile.y + i);
                coopMatLoad(matA[i], Ash, coordToOffset(si, sk, STRIDE_A_SH, false) / ELEMENTS_PER_VEC4, STRIDE_A_SH / ELEMENTS_PER_VEC4, A_LAYOUT);
            }

            coopmat<A_TYPE, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
                uint sj = lN * (C_COLS * warpInTile.x + j);
                // load B
                coopMatLoad(matB, Bsh, coordToOffset(sk, sj, STRIDE_B_SH, B_COL_MAJOR) / ELEMENTS_PER_VEC4, STRIDE_B_SH / ELEMENTS_PER_VEC4, B_LAYOUT);

                // do the matrix multiply for the current portion of the tile
                [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
                }
            }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        uint gi = TILE_M * tileID.y + lM * (C_ROWS * warpInTile.y + i);
        coopmat<C_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> matC[C_COLS];

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);
            coopMatLoad(matC[j], inputC.x, coordToOffset(gi, gj, strideC, false), strideC, C_LAYOUT);
        }

        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);

            result[i][j] = C_TYPE(alpha) * result[i][j] + C_TYPE(beta) * matC[j];
            coopMatStore(result[i][j], outputO.x, coordToOffset(gi, gj, strideD, false), strideD, D_LAYOUT);
        }
    }
}
