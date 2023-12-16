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

#define ITEMS_PER_THREAD 4

layout(set = 0, binding = 0) buffer Input{
    uint8_t in_data[][ITEMS_PER_THREAD];
};
layout(set = 0, binding = 1) buffer Output{
    uint32_t out_data[];
};
layout(set = 0, binding = 3) buffer Count{
    uint64_t out_count;
};
layout(set = 0, binding = 2) buffer Size{
    uint64_t size;
};

shared uint32_t shared_data[WORK_GROUP_SIZE];

layout(local_size_x = WORK_GROUP_SIZE, local_size_y = 1, local_size_z = 1)in;
void main(){
    uint global_id = uint(gl_GlobalInvocationID.x);
    uint local_id = uint(gl_LocalInvocationID.x);
    uint group_id = uint(gl_WorkGroupID.x);
    uint group_size = uint(WORK_GROUP_SIZE);

    uint8_t values_loaded[4] = in_data[local_id];
    uint8_t values_8[5];

    // TODO: find better way to load this.
    values_8[0] = (local_id * ITEMS_PER_THREAD + 0 < size) ? values_loaded[0] : uint8_t(0);
    values_8[1] = (local_id * ITEMS_PER_THREAD + 1 < size) ? values_loaded[1] : uint8_t(0);
    values_8[2] = (local_id * ITEMS_PER_THREAD + 2 < size) ? values_loaded[2] : uint8_t(0);
    values_8[3] = (local_id * ITEMS_PER_THREAD + 3 < size) ? values_loaded[3] : uint8_t(0);
    values_8[4] = uint8_t(0);
    
    
    uint32_t sum_local = 0;
    uint32_t values[5];
    
    // Unrolled exclusive scan
    for (uint32_t i = 0; i < 5; i++){
        uint32_t v = values_8[i];
        values[i] = sum_local;
        sum_local += v;
    }
    
    // Reduce using shared memory
    uint32_t si = local_id;
    shared_data[si] = 0;
    si += group_size;
    shared_data[si] = sum_local;

    uint32_t sum_group = sum_local;
    for (uint32_t offset = 1; offset < group_size; offset <<=1){
        memoryBarrierShared();
        barrier();
        sum_group = shared_data[si] + shared_data[si - offset];
        memoryBarrierShared();
        barrier();
        shared_data[si] = sum_group;
    }

    if (local_id == group_size - 1){
        out_count = sum_group;
    }


    sum_group -= sum_local;

    for (uint32_t i = 0; i < 5; ++i){
        values[i] += sum_group;
    }

    for (uint32_t i = 0; i < 4; ++i){
        if(values[i] != values[i+1]){
            out_data[uint(values[i])] = local_id * ITEMS_PER_THREAD + i;
        }
    }

}
