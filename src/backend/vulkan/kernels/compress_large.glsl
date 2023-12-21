#version 450
/*
    Ported from: https://github.com/mitsuba-renderer/drjit-core/blob/master/resources/compress.cuh

    Therefore licence: 
    
    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
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

#define ITEMS_PER_THREAD 16

// Defines weather to initialize outside of the buffer
// #define INIT

// Immutable buffer over input data
layout(set = 0, binding = 0) buffer Input{
    u32vec4 in_data[];
};
// Mutable buffer for output data
layout(set = 0, binding = 1) buffer Output{
    uint32_t out_data[];
};
layout(set = 0, binding = 2) buffer Count{
    uint32_t out_count;
};
// Buffer holding the `size` (number of elements) for the input buffer
layout(set = 0, binding = 3) buffer Size{
    uint32_t size;
};
// Atomic mutable scratch buffer, storing [index, 32 + item_size] u64s.
// The `index` part is used to atomically select partitions
// The other items are used to perform decoupled lookback (storing a flag and value)
layout(set = 0, binding = 4) buffer Scratch{
    uint64_t scratch[];
};
// We access the partition_counter using another view into the scratch_buffer
layout(set = 0, binding = 4) buffer PartitionCounter{
    uint32_t partition_counter;
};

// Idk. how we would have different views into the same shared data, therefore we use two shared variables.
shared uint32_t shared_data[WORK_GROUP_SIZE * ITEMS_PER_THREAD]; 

uint32_t clz(uint32_t x){
    return 31 - findMSB(x);
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
        shared_data[0] = uint32_t(atomicAdd(partition_counter, 1));
    }
    groupMemoryBarrier();
    barrier();
    partition_idx = subgroupBroadcast(shared_data[0], 0);
    groupMemoryBarrier();
    barrier();

    // We should now be able to replace block_idx with partition_idx in mitsuba's code

    
    uint8_t values_8[17];
    
    {
        u32vec4 tmp = in_data[partition_idx * block_size + thread_idx];
        values_8[0] = uint8_t (tmp.x >> 0);
        values_8[1] = uint8_t (tmp.x >> 8);
        values_8[2] = uint8_t (tmp.x >> 16);
        values_8[3] = uint8_t (tmp.x >> 24);
        values_8[4] = uint8_t (tmp.y >> 0);
        values_8[5] = uint8_t (tmp.y >> 8);
        values_8[6] = uint8_t (tmp.y >> 16);
        values_8[7] = uint8_t (tmp.y >> 24);
        values_8[8] = uint8_t (tmp.z >> 0);
        values_8[9] = uint8_t (tmp.z >> 8);
        values_8[10] = uint8_t (tmp.z >> 16);
        values_8[11] = uint8_t (tmp.z >> 24);
        values_8[12] = uint8_t (tmp.w >> 0);
        values_8[13] = uint8_t (tmp.w >> 8);
        values_8[14] = uint8_t (tmp.w >> 16);
        values_8[15] = uint8_t (tmp.w >> 24);

        values_8[16] = uint8_t(0);
    }
    
    // Unrolled exclusive scan
    uint32_t sum_local = 0;
    uint32_t values[17];
    for (uint i = 0; i < 17; i++){
        uint32_t v = values_8[i];
        values[i] = sum_local;
        sum_local += v;
    }

    // Block-level reduction of partial sum over 16 elements via shared memory
    uint32_t si = thread_idx;
    shared_data[si] = 0;
    si += block_size;
    shared_data[si] = sum_local;

    uint32_t sum_block = sum_local;
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
        atomicStore(scratch[scratch_idx], (uint64_t(sum_block) << 32) | 1ul, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    }

    uint32_t prefix = 0;

    // Each thread looks back a different amount
    int32_t shift = int32_t(lane) - int32_t(warp_size);

    // Decoupled lookback iteration
    /* Compute prefix due to previous blocks using warp-level primitives.
       Based on "Single-pass Parallel Prefix Scan with Decoupled Look-back"
       by Duane Merrill and Michael Garland */
    while(true){
        uint64_t tmp = atomicLoad(scratch[scratch_idx + shift], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
        uint32_t flag = uint32_t(tmp);

        // Retry if at least one of the predecessors hasn't made any progress yet
        if(subgroupAny(flag == 0)){
            continue;
        }

        uint32_t mask = subgroupBallot(flag == 2).x; 
        uint32_t value = uint32_t(tmp >> 32);

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
        atomicStore(scratch[scratch_idx], (uint64_t(sum_block) << 32) | 2ul, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);

        if(partition_idx == grid_size -1){
            out_count = sum_block;
        }
    }

    sum_block -= sum_local;
    for (uint i = 0; i < 17; i++){
        values[i] += sum_block;
    }

    for (uint i = 0; i < 16; i++){
        if(values[i] != values[i+1]){
            out_data[uint(values[i])] = (partition_idx * block_size + thread_idx) * 16 + i;
        }
    }
}
