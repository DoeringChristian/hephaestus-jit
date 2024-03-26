#version 450

// Explicit arythmetic types
#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require


// TODO: Optimization

layout(set = 0, binding = 0) buffer Input{
    TYPE in_data[];
};
layout(set = 0, binding = 1) buffer Output{
    TYPE out_data[];
};
layout(set = 0, binding = 2) buffer Size{
    uint64_t size;
};

shared TYPE shared_data[WORK_GROUP_SIZE];

layout(local_size_x = WORK_GROUP_SIZE, local_size_y = 1, local_size_z = 1)in;
void main(){
    uint global_id = uint(gl_GlobalInvocationID.x);
    uint local_id = uint(gl_LocalInvocationID.x);
    uint group_id = uint(gl_WorkGroupID.x);
    uint group_size = uint(WORK_GROUP_SIZE);

    shared_data[local_id] = global_id < size ? in_data[global_id]: INIT;
    
    memoryBarrierShared();
    barrier();

    for (uint s = group_size/2; s > 0; s>>=1){
        if (local_id < s){
            TYPE a = shared_data[local_id];
            TYPE b = shared_data[local_id + s];
            TYPE result = REDUCE;
            shared_data[local_id] = result;
        }
    }

    if (local_id == 0) out_data[group_id] = shared_data[0];
}
