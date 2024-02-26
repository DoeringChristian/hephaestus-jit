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


/*
#define INDEX
#define TYPE
#define WORKGROUP_SIZE
*/

layout(set = 0, binding = 0) buffer Value{
    TYPE value[];
};
layout(set = 0, binding = 1) buffer Divisor{
    TYPE divisor[];
};
layout(set = 0, binding = 2) buffer Output{
    TYPE output[];
};


layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main(){
    uint i = uint(gl_GlobalInvocationID.x);
    TYPE div = divisor[i];
    TYPE val = value[i];
    output[i] = (val + div - 1) / div;
}
