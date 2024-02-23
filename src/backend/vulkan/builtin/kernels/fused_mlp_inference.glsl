/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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


/*
#define WIDTH
#define N_ITERS
#define OUT_T
// #define ACTIVATION
#define INFERENCE
#define BACKWARD
// #define OUTPUT
#define A_BITS
*/

// Compatibility macros
#define LAYOUT_ROW_MAJOR gl_CooperativeMatrixLayoutRowMajor
#define LAYOUT_COL_MAJOR gl_CooperativeMatrixLayoutColumnMajor
#define threadIdx uvec3(gl_LocalInvocationID)
#define blockIdx uvec3(gl_WorkGroupID)


// Arguments: 
//
// `input` points to the input matrix. Can be any width.
// `weights` points to the weight matrices (contiguous in memory).
// `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
// `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)
layout(set = 0, binding = 0) buffer Input{
    uvec4 input_uvec4[];
};
layout(set = 0, binding = 1) buffer Weights{
    float16_t weights[];
};
layout(set = 0, binding = 2) buffer Out{
    uvec4 output_uvec4[];
};
layout(set = 0, binding = 3) buffer Config{
    uint32_t output_stride;
    uint32_t batch_size;
    uint32_t in_width;
    uint32_t n_hidden_matmuls;
    int32_t input_layout;
    int32_t output_layout;
};

// Shared memory contains the intermediate activations of blockDim.y*16 elements.
// In some cases, it also contains the weight matrix for the first and last layer.


const uint SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
const uint ELEMENTS_PER_VEC4 = 16/(A_BITS / 8);// 16 bytes, A_BITS bits per element
const uint SHMEM_ELEMENTS = (16 + 16 * N_ITERS) * (WIDTH * SKEW);
shared uvec4 shmem[SHMEM_ELEMENTS / ELEMENTS_PER_VEC4]; // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*N_ITERS rows of intermediate activations

const uint N_BLOCK_ROWS = WIDTH/16;
layout(local_size_x = 32, local_size_y = N_BLOCK_ROWS, local_size_z = 1) in;

// Define ReLU
#ifndef ACTIVATION
#define ACTIVATION RELU
#endif

float16_t warp_activation(float16_t frag){
#if ACTIVATION == RELU
    return max(frag, float16_t(0));
#endif
}

// Port of: https://github.com/NVlabs/tiny-cuda-nn/blob/235d1fde956dc04966940f9d1bec66aa3bdb705a/src/fully_fused_mlp.cu#L47
void threadblock_layer(uint weights_this_layer, uint out_intermediate_threadblock_this_layer, uint activation_aux){
    // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	//           Can be forward activations or backward activations, depending on caller.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.

    // const uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    const uint32_t N_BLOCKS = WIDTH / 16;

    // If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
    // const int weights_layout_t = BACKWARD ? LAYOUT_ROW_MAJOR : LAYOUT_COL_MAJOR;
    const int weights_layout_t = LAYOUT_COL_MAJOR;

    // Fragments
    // NOTE: using transposed form? $H'_{i+1}^T = H_i^T \cdot W_i^T$
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> act_frag;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> weights_frag[N_BLOCKS];
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> result_frag[N_ITERS];

    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

    const uint32_t weights_col = 16 * wi;

    barrier();

    // Load N_BLOCKS chunks of weights from global memory into registers.
    [[unroll]] for (uint i = 0; i < N_BLOCKS; i++){
        coopMatLoad(weights_frag[i], weights, weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH, weights_layout_t);
    }

    [[unroll]] for (uint l = 0; l < N_ITERS; l++){
        result_frag[l] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

        [[unroll]] for (uint i = 0; i < N_BLOCKS; i++){
            // Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
            // NOTE: we can't cast shmem therefore we need to change the index by dividing by (elems in uvec4)
            // WARN: `elem_idx` has to be divisble by `ELEMENTS_PER_VEC4`
            uint elem_idx = 16 * i + (16 * l) * (WIDTH + SKEW);
            coopMatLoad(act_frag, shmem, elem_idx/ELEMENTS_PER_VEC4, WIDTH + SKEW, LAYOUT_ROW_MAJOR);
            result_frag[l] = coopMatMulAdd(act_frag, weights_frag[i], result_frag[l]);
        }

		// Activation
        for (uint i = 0; i < result_frag[l].length(); i++){
            result_frag[l][i] = warp_activation(result_frag[l][i]);
        }
    }

    barrier();

    [[unroll]] for (uint l = 0; l < N_ITERS; l++){
        // NOTE: index by u32x4 (weights_col is divisible by 16)
        uint elem_idx = weights_col + l * 16 * (WIDTH + SKEW);
        coopMatStore(result_frag[l], shmem, elem_idx/ELEMENTS_PER_VEC4, WIDTH + SKEW, LAYOUT_ROW_MAJOR);
    }
}

void threadblock_load_input_static(uint input_threadblock){
    // act_shmem will be filled by the thread block's chunk of input_threadblock

    // const uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    
    // Indices
    uint32_t li = threadIdx.x; // index in warp ("lane index")
    uint32_t wi = threadIdx.y; // index in block ("warp index")
    
    uint32_t lane_offset = (8 * li) % WIDTH;
    uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

    [[unroll]] for (uint i = 0; i < N_ITERS; i++){
        // WARN: both `elem_idx_shmem` and `elem_idx_input` have to be divisible by ELEMENTS_PER_VEC4
        uint elem_idx_shmem = lane_offset + (row + 16 * i) * (WIDTH + SKEW);
        uint elem_idx_input = input_threadblock + lane_offset + (row + 16 * i) * WIDTH;

        shmem[elem_idx_shmem / ELEMENTS_PER_VEC4] = input_uvec4[elem_idx_input / ELEMENTS_PER_VEC4];
    }
}

void threadblock_write_output_static(uint output_threadblock){
    // output_threadblock will be filled by the thread block's act_shmem
    
    // const uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
    
    // Indices
	uint32_t li = threadIdx.x; // index in warp ("lane index")
	uint32_t wi = threadIdx.y; // index in block ("warp index")

    uint32_t lane_offset = (ELEMENTS_PER_VEC4 * li) % WIDTH;
	uint32_t row = (ELEMENTS_PER_VEC4 * li + wi * ELEMENTS_PER_VEC4 * 32) / WIDTH;

    barrier();

    [[unroll]] for (uint i = 0; i < N_ITERS; i++){
        // WARN: both `elem_idx_output` and `elem_idx_shmem` have to be divisible by ELEMENTS_PER_VEC4
        uint elem_idx_output = output_threadblock + lane_offset + (row + 16 * i) * WIDTH;
        uint elem_idx_shmem = lane_offset + (row + 16 * i) * (WIDTH + SKEW);

        output_uvec4[elem_idx_output / ELEMENTS_PER_VEC4] = shmem[elem_idx_shmem / ELEMENTS_PER_VEC4];
    }
}

void main(){
    // Each block computes exactly one 16-element chunk of the batch.
    uint32_t elem_idx = 16 * blockIdx.x * N_ITERS;
    
    // First layer
    if (input_layout == LAYOUT_COL_MAJOR || in_width != WIDTH){
        if (input_layout == LAYOUT_ROW_MAJOR){
            // TODO: https://github.com/NVlabs/tiny-cuda-nn/blob/235d1fde956dc04966940f9d1bec66aa3bdb705a/src/fully_fused_mlp.cu#L524
        }else{
            // TODO: https://github.com/NVlabs/tiny-cuda-nn/blob/235d1fde956dc04966940f9d1bec66aa3bdb705a/src/fully_fused_mlp.cu#L526
        }
    }else{
        // If the input has the same width & layout as the hidden layers, we can simply use the network's regular layer routine (with static size)
		// instead of using the slower dynamic input layer routine.
        threadblock_load_input_static(elem_idx * WIDTH);
        threadblock_layer(0, elem_idx * WIDTH, 0);
    }

    uint32_t first_weights_stride = WIDTH * in_width;
	uint32_t weights_stride = WIDTH * WIDTH;
	uint32_t layer_stride = WIDTH * batch_size;
    
    // Hidden layers

    for (uint k = 0; k < n_hidden_matmuls; k++){
        threadblock_layer(first_weights_stride + weights_stride * k, layer_stride * (k + 1) + elem_idx * WIDTH, 0);
    }

    threadblock_write_output_static(elem_idx * WIDTH);
}

