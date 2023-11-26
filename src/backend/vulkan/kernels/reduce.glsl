#extension GL_EXT_shader_explicit_arithmetic_types: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

layout(set = 0, binding = 0) buffer Src{
    {{TYPE}} src[];
};
layout(set = 0, binding = 1) buffer Dst{
    {{TYPE}} dst;
};
layout(set = 0, binding = 2) buffer Size{
    uint64_t size;
};

shared {{TYPE}} shared_memory[{{WORK_GROUP_SIZE}}];

layout(local_size_x = {{WORK_GROUP_SIZE}}, local_size_y = 1, local_size_z = 1)in;
void main(){
    uint local_id = uint(gl_LocalInvocationID.x);
    uint group_id = uint(gl_WorkGroupID.x);
    uint group_size = uint({{WORK_GROUP_SIZE}});
}
