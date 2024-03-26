#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require

struct Instance{
    float32_t transform[12];
    uint32_t geometry;
};
struct VkInstance{
    float32_t transform[12];
    uint32_t instance_custom_index_and_mask;
    uint32_t instanec_sbt_offset_and_flags;
    uint64_t acceleration_structure_reference;
};

layout(set = 0, binding = 0) buffer Src{
    Instance src_instances[];
};
layout(set = 0, binding = 1) buffer References{
    uint64_t src_refs[];
};
layout(set = 0, binding = 2) buffer Dst{
    VkInstance dst_instances[];
};


void main(){
    uint idx = uint(gl_GlobalInvocationID.x);
    
    Instance src_instance = src_instances[idx];

    VkInstance dst_instance;
    dst_instance.transform = src_instance.transform;
    dst_instance.instance_custom_index_and_mask = uint32_t(0xff000000);
    dst_instance.instanec_sbt_offset_and_flags = uint32_t(0x01000000);
    dst_instance.acceleration_structure_reference = src_refs[src_instance.geometry];
    
    dst_instances[idx] = dst_instance;
}
