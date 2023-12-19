#version 450
/*
    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file
*/



#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

#extension GL_KHR_memory_scope_semantics: require

#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_vote: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_KHR_shader_subgroup_shuffle: require
#extension GL_KHR_shader_subgroup_shuffle_relative: require

#define INCLUSIVE

// N: Number of vectors to load
// M: Number of element per vector 
#define N 4
#define M 4


// TODO: Optimization

layout(set = 0, binding = 0) buffer Input{
    u32vec4 in_data[];
};
layout(set = 0, binding = 1) buffer Output{
    u32vec4 out_data[];
};
layout(set = 0, binding = 1) buffer Outputu32{
    uint32_t out_data_u32[];
};
layout(set = 0, binding = 2) buffer Size{
    uint32_t size;
};
layout(set = 0, binding = 3) buffer Scratch{
    uint64_t scratch[];
};
layout(set = 0, binding = 4) buffer Index{
    uint32_t global_index;
};

shared uint32_t shared_data[WORK_GROUP_SIZE * N * M]; 

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
        shared_data[0].x = atomicAdd(global_index, 1);
    }
    groupMemoryBarrier();
    barrier();
    partition_idx = subgroupBroadcast(shared_data[0].x, 0);
    groupMemoryBarrier();
    barrier();

    // We should now be able to replace block_idx with partition_idx in mitsuba's code

    u32vec4 v[N];

    for (uint i = 0; i < N; i++){
        uint32_t j = (partition_idx * N + i) * block_size + thread_idx;

        u32vec4 value = in_data[j];

        // TODO: add bound check (especially for floats)!

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
    uint32_t values[N * M];
    for (uint i = 0; i < N; i++){
        values[i * M +0] = shared_data[(thread_idx * N + i) * M + 0];
        values[i * M +1] = shared_data[(thread_idx * N + i) * M + 1];
        values[i * M +2] = shared_data[(thread_idx * N + i) * M + 2];
        values[i * M +3] = shared_data[(thread_idx * N + i) * M + 3];
    }

    // Unroled exclusive prefix sum
    uint32_t sum_local = uint32_t(0);
    for (uint i = 0; i < N * M; i++){
        uint32_t v = values[i];

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
    uint scratch_idx = partition_idx + warp_size; // scratch buffer is offset by warp_size to not cause deadlocks for first thread
    if (thread_idx == block_size - 1){
        atomicStore(scratch[scratch_idx], (uint64_t(sum_block) << 32) | 1ul, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    }

    uint32_t prefix = uint32_t(0);

    // Each thread looks back a different amount
    int32_t shift = int32_t(lane) - int32_t(warp_size);


    uint debug_i = 0;

    // Decoupled lookback iteration
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

        u32vec4 v; 
        v.x = shared_data[j * M + 0];
        v.y = shared_data[j * M + 1];
        v.z = shared_data[j * M + 2];
        v.w = shared_data[j * M + 3];
        uint index_out = j + partition_idx * (N * block_size);
        
        out_data[j + partition_idx * (N * block_size)] = v;
    }
}
