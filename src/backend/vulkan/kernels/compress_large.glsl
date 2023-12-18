#version 450
/*
    Ported from: https://github.com/mitsuba-renderer/drjit-core/blob/master/resources/compress.cuh

    Therefore licence: 
    
    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
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


#define ITEMS_PER_THREAD 16

layout(set = 0, binding = 0) buffer Input{
    u32vec4 in_data[];
};
layout(set = 0, binding = 1) buffer Output{
    uint32_t out_data[];
};
layout(set = 0, binding = 2) buffer Scratch{
    uint64_t scratch[];
};
layout(set = 0, binding = 3) buffer Count{
    uint32_t out_count;
};
// layout(set = 0, binding = 4) buffer Size{
//     uint32_t size;
// };

shared uint32_t shared_data[WORK_GROUP_SIZE * ITEMS_PER_THREAD];

uint32_t clz(uint32_t x){
    return 32 - findMSB(x);
}

layout(local_size_x = WORK_GROUP_SIZE, local_size_y = 1, local_size_z = 1)in;
void main(){
    uint global_id = uint(gl_GlobalInvocationID.x);
    uint local_id = uint(gl_LocalInvocationID.x);
    uint group_id = uint(gl_WorkGroupID.x);
    uint group_size = uint(WORK_GROUP_SIZE);
    uint n_groups = uint(gl_NumWorkGroups.x);
    uint subgroup_size = uint(gl_SubgroupSize);
    uint subgroup_id = uint(gl_SubgroupID);

    uint scratch_idx = 32;

    // Load 16 elements
    uint8_t values_8[17];

    u32vec4 loaded = in_data[group_id * group_size + local_id];

    values_8[0] = uint8_t(loaded.x >> 0);
    values_8[1] = uint8_t(loaded.x >> 8);
    values_8[2] = uint8_t(loaded.x >> 16);
    values_8[3] = uint8_t(loaded.x >> 24);
    values_8[4] = uint8_t(loaded.y >> 0);
    values_8[5] = uint8_t(loaded.y >> 8);
    values_8[6] = uint8_t(loaded.y >> 16);
    values_8[7] = uint8_t(loaded.y >> 24);
    values_8[8] = uint8_t(loaded.z >> 0);
    values_8[9] = uint8_t(loaded.z >> 8);
    values_8[10] = uint8_t(loaded.z >> 16);
    values_8[11] = uint8_t(loaded.z >> 24);
    values_8[12] = uint8_t(loaded.w >> 0);
    values_8[13] = uint8_t(loaded.w >> 8);
    values_8[14] = uint8_t(loaded.w >> 16);
    values_8[15] = uint8_t(loaded.w >> 24);

    values_8[16] = uint8_t(0);
    
    // Unrolled exclusive scan
    uint32_t sum_local = 0;
    uint32_t values[17];
    for (uint i = 0; i < 17; ++i){
        uint32_t v = values_8[i];
        values[i] = sum_local;
        sum_local += v;
    } 
    
    // Block-level reduction of partial sum over 16 elements via shared memory
    uint32_t si = local_id;
    shared_data[si] = 0;
    si += group_size;
    shared_data[si] = sum_local;

    uint32_t sum_block = sum_local;
    for (uint offset = 1; offset < group_size; offset <<= 1){
        memoryBarrierShared();
        barrier();
        sum_block = shared_data[si] + shared_data[si - offset];
        memoryBarrierShared();
        barrier();
        shared_data[si] = sum_block;
    }

    // Store block-level partial inclusive scan value in global memory
    scratch_idx += group_id;
    if(local_id == group_size -1){
        atomicStore(scratch[scratch_idx], (uint64_t(sum_block) << 32) | 1ul, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    }

    uint32_t lane = local_id & (subgroup_size - 1);
    uint32_t prefix = 0;
    int32_t shift = int32_t(lane) - int32_t(subgroup_size);

    /* Compute prefix due to previous blocks using warp-level primitives.
       Based on "Single-pass Parallel Prefix Scan with Decoupled Look-back"
       by Duane Merrill and Michael Garland */
    while(true){
        /// Prevent loop invariant code motion of loads
        uint64_t tmp = atomicLoad(scratch[scratch_idx + shift], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
        uint32_t flag = uint32_t(tmp);

        if(subgroupAny(flag == 0)){
            continue;
        }

        uint32_t mask = subgroupBallot(flag == 2).x; // Hope that it gets stored in x
        uint32_t value = uint32_t(tmp >> 32);

        if (mask == 0){
            prefix += value;
            shift -= int32_t(subgroup_size);
        }else{
            uint32_t index = 31 - clz(mask);
            if (lane >= index){
                prefix += value;
            }
            break;
        }
    }


    // Warp-level (subgroup-level) reduction
    for (uint offset = 16; offset > 0; offset /= 2){
        prefix += subgroupShuffleDown(prefix, offset);
    }
    sum_block += subgroupShuffle(prefix, 0);

    // Store block-level complete inclusive scan value in global memory
    if (local_id == group_size - 1){
        atomicStore(scratch[scratch_idx], (uint64_t(sum_block) << 32) | 2ul, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);

        if(group_id == n_groups - 1){
            out_count = sum_block;
        }
    }

    sum_block -= sum_local;
    for (uint i = 0; i < 17; ++i){
        values[i] += sum_block;
    }

    for (uint i = 0; i < 16; ++i){
        if(values[i] != values[i+1]){
            out_data[values[i]] = (group_id * group_size + local_id) * 16 + i;
        }
    }
}
