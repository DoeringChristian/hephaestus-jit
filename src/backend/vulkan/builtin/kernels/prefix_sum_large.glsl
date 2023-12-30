#version 450
/*
    This implementation is both based on Mitsuba's prefix sum.
    In order to make it more compatible, I added the threadlock prevention strategy described
    here: https://github.com/b0nes164/GPUPrefixSums

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file
*/



// Explicit arythmetic types
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

// Atomics and Memory
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_atomic_int64: require

// Subgroup
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_vote: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_KHR_shader_subgroup_shuffle: require
#extension GL_KHR_shader_subgroup_shuffle_relative: require
#extension GL_EXT_shader_subgroup_extended_types_int64: require


// N: Number of vectors to load
// M: Number of element per vector 
#ifndef N
#define N 4
#endif

#ifndef M
#define M 4
#endif

// Defines if inclusive or exclusive sum is performed
// #define INCLUSIVE

// Defines if initialization is required
// #define INIT

// Defines the type
#ifndef T
#define T uint32_t
#endif
// Defines the vector type used for loading must be `{short form of T}vec{M}`
#ifndef VT
#define VT u32vec4
#endif


// TODO: Optimization

// Immutable buffer over input data
layout(set = 0, binding = 0) buffer Input{
    VT in_data[];
};
// Mutable buffer for output data
layout(set = 0, binding = 1) buffer Output{
    VT out_data[];
};
// DEBUG:
layout(set = 0, binding = 1) buffer Outputu32{
    uint32_t out_data_u32[];
};
// Buffer holding the `size` (number of elements) for the input buffer
layout(set = 0, binding = 2) buffer Size{
    uint32_t size;
};
// Atomic mutable scratch buffer, storing [index, 32 + item_size] u64s.
// The `index` part is used to atomically select partitions
// The other items are used to perform decoupled lookback (storing a flag and value)
layout(set = 0, binding = 3) buffer Scratch{
    uint64_t scratch[];
};
// We access the partition_counter using another view into the scratch_buffer
layout(set = 0, binding = 3) buffer PartitionCounter{
    uint32_t partition_counter;
};

// Idk. how we would have different views into the same shared data, therefore we use two shared variables.
shared uint32_t shared_partition_index; 
shared T shared_data[WORK_GROUP_SIZE * N * M]; 

uint32_t clz(uint32_t x){
    return 31 - findMSB(x);
}

/*
 * Combine a value and flag in one uint64_t value.
 * Ported from Mitsuba
 *
 * The Vulkan backend implementation for *large* numeric types (double precision
 * floats, 64 bit integers) has the following technical limitation: when
 * reducing 64-bit integers, their values must be smaller than 2**62. When
 * reducing double precision arrays, the two least significant mantissa bits
 * are clamped to zero when forwarding the prefix from one 512-wide block to
 * the next (at a very minor loss in accuracy). The reason is that the
 * operations requires two status bits (in the `flags` variable)  to coordinate
 * the prefix and status of each 512-wide block, and those must each fit into a
 * single 64 bit value (128-bit writes aren't guaranteed to be atomic).
*/
uint64_t combine(T value, uint32_t flag){
    uint64_t combined;
    
    // TODO: add more types
#if T == uint32_t
    combined = uint64_t(value) << 32 | flag;
#elif T == float32_t
    combined = uint64_t(floatBitsToUint(value)) << 32 | flag;
#elif T == uint64_t 
    combined = (value << 2) | flag;
#elif T == float64_t 
    combined = (floatBitsToUint(value) & ~3ul) | flag;
#endif

    return combined;
}

/*
 * Extract a value and flag from one uint64_t value.
 * Ported from Mitsuba
 *
 * The Vulkan backend implementation for *large* numeric types (double precision
 * floats, 64 bit integers) has the following technical limitation: when
 * reducing 64-bit integers, their values must be smaller than 2**62. When
 * reducing double precision arrays, the two least significant mantissa bits
 * are clamped to zero when forwarding the prefix from one 512-wide block to
 * the next (at a very minor loss in accuracy). The reason is that the
 * operations requires two status bits (in the `flags` variable)  to coordinate
 * the prefix and status of each 512-wide block, and those must each fit into a
 * single 64 bit value (128-bit writes aren't guaranteed to be atomic).
*/
void extract(in uint64_t src, out T value, out uint32_t flag){
    
    // TODO: add more types
#if T == uint32_t
    flag = uint32_t(src);
    value = uint32_t(src >> 32);
#elif T == float32_t
    flag = uint32_t(src);
    value = floatBitsToUint(uint32_t(src >> 32));
#elif T == uint64_t 
    flags = uint32_t(src) & 3u;
    value = src >> 2;
#elif T == float64_t 
    flags = uint32_t(src) & 3u;
    value = floatBitsToUint(src & ~32ul);
#endif

}

layout(local_size_x = WORK_GROUP_SIZE, local_size_y = 1, local_size_z = 1)in;
void main(){
    uint thread_idx = uint(gl_LocalInvocationID.x);
    uint block_idx = uint(gl_WorkGroupID.x);
    uint block_size = uint(WORK_GROUP_SIZE);
    uint global_thread_idx = uint(gl_GlobalInvocationID.x);
    uint grid_size = uint(gl_NumWorkGroups.x);
    uint warp_count = uint(gl_NumSubgroups);
    uint warp_size = uint(gl_SubgroupSize);
    uint warp_idx = uint(gl_SubgroupID);
    uint lane = uint(gl_SubgroupInvocationID);

    // Atomically aquire partition index
    uint32_t partition_idx = 0;
    if (warp_idx == 0 && lane == 0){
        shared_partition_index = uint32_t(atomicAdd(partition_counter, 1));
    }
    groupMemoryBarrier();
    barrier();
    partition_idx = subgroupBroadcast(shared_partition_index, 0);
    groupMemoryBarrier();
    barrier();

    // We should now be able to replace block_idx with partition_idx in mitsuba's code

    VT v[N];

    for (uint i = 0; i < N; i++){
        uint32_t j = (partition_idx * N + i) * block_size + thread_idx;

        VT value = in_data[j];

        #ifdef INIT
        // TODO: test for conditions where M < 4
        j *= M;
        if(j + 0 >= size) value.x = 0;
        if(j + 1 >= size) value.y = 0;
        if(j + 2 >= size) value.z = 0;
        if(j + 3 >= size) value.w = 0;
        #endif

        v[i] = value;
    }

    for (uint i = 0; i < N; i++){
        shared_data[(i * block_size + thread_idx) * M + 0] = v[i].x;
        shared_data[(i * block_size + thread_idx) * M + 1] = v[i].y;
        shared_data[(i * block_size + thread_idx) * M + 2] = v[i].z;
        shared_data[(i * block_size + thread_idx) * M + 3] = v[i].w;
    }

    memoryBarrierShared();
    barrier();

    // Fetch input from shared memory
    T values[N * M];
    for (uint i = 0; i < N; i++){
        values[i * M +0] = shared_data[(thread_idx * N + i) * M + 0];
        values[i * M +1] = shared_data[(thread_idx * N + i) * M + 1];
        values[i * M +2] = shared_data[(thread_idx * N + i) * M + 2];
        values[i * M +3] = shared_data[(thread_idx * N + i) * M + 3];
    }

    // Unroled exclusive prefix sum
    T sum_local = T(0);
    for (uint i = 0; i < N * M; i++){
        T v = values[i];

        #ifdef INCLUSIVE
        sum_local += v;
        values[i] = sum_local;
        #else
        values[i] = sum_local;
        sum_local += v;
        #endif
        
    }
    
    memoryBarrierShared();
    barrier();
    
    // Block-level inclusive prefix sum of 'sum_local' via shared memory
    uint si = thread_idx;
    shared_data[si] = 0;
    si += block_size;

    T sum_block = sum_local;
    for (uint offset = 1; offset < block_size; offset <<= 1){
        shared_data[si] = sum_block;
        memoryBarrierShared();
        barrier();
        sum_block = shared_data[si] + shared_data[si - offset];
        memoryBarrierShared();
        barrier();
    }

    // Store tentative block-level inclusive prefix sum value in global memory
    // (still missing prefix from predecessors)
    // uint scratch_idx = partition_idx + SCRATCH_OFFSET;
    
    // scratch buffer is offset by warp_size to not cause deadlocks for first thread
    // Also by the atomic counter
    
    uint scratch_idx = partition_idx + warp_size + 1;
    if (thread_idx == block_size - 1){
        uint64_t combined = combine(sum_block, 1u);
        atomicStore(scratch[scratch_idx], combined, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    }

    T prefix = T(0);

    // Each thread looks back a different amount
    int32_t shift = int32_t(lane) - int32_t(warp_size);


    // Decoupled lookback iteration
    while(true){
        uint64_t tmp = atomicLoad(scratch[scratch_idx + shift], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
        // uint32_t flag = uint32_t(tmp);
        uint32_t flag;
        T value;
        extract(tmp, value, flag);
        
        // Retry if at least one of the predecessors hasn't made any progress yet
        if(subgroupAny(flag == 0)){
            continue;
        }

        uint32_t mask = subgroupBallot(flag == 2).x; 
        // uint32_t value = uint32_t(tmp >> 32);
        
        if (mask == 0){
            prefix += value;
            shift -= int32_t(warp_size);
        }else{
            // Lane 'index' is done!
            uint32_t index = 31 - clz(mask);

            // Sum up all the unconverged (higher) lanes *and* 'index'
            if (lane >= index){
                prefix += value;
            }
            break;
        }
    }
    
    // Warp-level sum reduction of 'prefix'
    for (uint offset = warp_size / 2; offset > 0; offset /= 2){
        prefix += subgroupShuffleDown(prefix, offset);
    }
    
    // Broadcast the reduced 'prefix' value from lane 0
    prefix = subgroupBroadcast(prefix, 0);
    
    // Offset the local block sum with the final prefix
    sum_block += prefix;
    
    
    // Store block-level complete inclusive prefixnsum value in global memory
    if(thread_idx == block_size - 1){
        uint64_t combined = combine(sum_block, 2u);
        atomicStore(scratch[scratch_idx], combined, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    }

    sum_block -= sum_local;
    for (uint i = 0; i < N * M; i++){
        values[i] += sum_block;
    }
    

    // Store input into shared memory
    for (uint i = 0; i < N; i++){
        shared_data[(thread_idx * N + i) * M + 0] = values[N * i + 0];
        shared_data[(thread_idx * N + i) * M + 1] = values[N * i + 1];
        shared_data[(thread_idx * N + i) * M + 2] = values[N * i + 2];
        shared_data[(thread_idx * N + i) * M + 3] = values[N * i + 3];
    }
    
    memoryBarrierShared();
    barrier();

    // Copy shared memory back to global memory
    for (uint i = 0; i < N; i++){
        uint j = i * block_size + thread_idx;

        VT v; 
        v.x = shared_data[j * M + 0];
        v.y = shared_data[j * M + 1];
        v.z = shared_data[j * M + 2];
        v.w = shared_data[j * M + 3];
        uint index_out = j + partition_idx * (N * block_size);
        
        out_data[j + partition_idx * (N * block_size)] = v;
    }
}
