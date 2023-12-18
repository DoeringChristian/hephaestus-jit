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


// TODO: Optimization

layout(set = 0, binding = 0) buffer Scratch{
    uint64_t scratch[];
};
layout(set = 0, binding = 1) buffer Size{
    uint32_t size;
};

layout(local_size_x = WORK_GROUP_SIZE, local_size_y = 1, local_size_z = 1)in;
void main(){
    uint global_id = uint(gl_GlobalInvocationID.x);
    uint local_id = uint(gl_LocalInvocationID.x);
    uint group_id = uint(gl_WorkGroupID.x);
    uint group_size = uint(WORK_GROUP_SIZE);
    uint n_groups = uint(gl_NumWorkGroups.x);

    for(uint i = global_id; i < size; i += group_size * n_groups){
        scratch[i] = ( i < 32 ) ? 2 : 0;
    }

}
