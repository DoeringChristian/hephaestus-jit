#version 450

layout(set = 0, binding = 0) buffer Config{
    uint width;
    uint height;
    uint src_pitch;
    uint dst_pitch;
    uint src_offset;
    uint dst_offset;
};
layout(set = 0, binding = 1) buffer Src{
    uint src[];
};
layout(set = 0, binding = 2) buffer Dst{
    uint dst[];
};


void main(){
    uint i = uint(gl_GlobalInvocationID.x);
    uint x = i % width;
    uint y = i / width;

    uint i_src = src_pitch * y + x + src_offset;
    uint i_dst = dst_pitch * y + x + dst_offset;

    dst[i_dst] = src[i_src];
}
