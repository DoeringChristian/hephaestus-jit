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

// Subgroup
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_vote: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_KHR_shader_subgroup_shuffle: require
#extension GL_KHR_shader_subgroup_shuffle_relative: require

// TODO: Optimization

layout(set = 0, binding = 0) buffer Scratch{
    uint64_t scratch[];
};
layout(set = 0, binding = 1) buffer Size{
    uint32_t size;
};

/*
Initialize the scratch buffer for large prefix sums.
The first element of the scratch buffer is a atomic counter used to get partition indices.
The next 32 elements have to be initialized with flag == 2 for the lookback algorithm.
All other elements have to be initialized with 0.
*/
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

    for(uint i = global_thread_idx; i < size; i += block_size * grid_size){
        // WARN: using `warp_size` here means that the buffer has to be
        // initialized with the same device that uses it.
        scratch[i] = ( 1 <= i && i < warp_size + 1 ) ? 2 : 0;
    }

}
