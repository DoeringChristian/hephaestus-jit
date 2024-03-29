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

// Atomics and Memory
#extension GL_KHR_memory_scope_semantics: require
#extension GL_EXT_shader_atomic_int64: require

// Subgroup
#extension GL_KHR_shader_subgroup_basic: require
#extension GL_KHR_shader_subgroup_vote: require
#extension GL_KHR_shader_subgroup_ballot: require
#extension GL_KHR_shader_subgroup_shuffle: require
#extension GL_KHR_shader_subgroup_shuffle_relative: require
#extension GL_EXT_shader_subgroup_extended_types_int64: require

#extension GL_KHR_cooperative_matrix: require

#ifndef COOP_M
#define COOP_M 16
#endif

#ifndef COOP_K
#define COOP_K 16
#endif

#ifndef COOP_N
#define COOP_N 16
#endif

#ifndef T
#define T float32_t
#endif

#ifndef TRANSPOSE_B
#define TRANSPOSE_B false
#endif

layout(set = 0, binding = 0) buffer Config{
    uint32_t N;
    uint32_t M;
    uint32_t K;
};

layout(set = 0, binding = 1) buffer MatA{
    T data_a[];
};
layout(set = 0, binding = 2) buffer MatB{
    T data_b[];
};
layout(set = 0, binding = 3) buffer MatC{
    T data_c[];
};

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1)in;
void main(){
    // Get a global id of the current subgroup
    uint gsid = gl_SubgroupID + gl_WorkGroupID.x * gl_NumSubgroups;

    coopmat<T, gl_ScopeSubgroup, COOP_N, COOP_K, gl_MatrixUseA> coop_a;
    coopmat<T, gl_ScopeSubgroup, COOP_K, COOP_M, gl_MatrixUseB> coop_b;
    coopmat<T, gl_ScopeSubgroup, COOP_K, COOP_K, gl_MatrixUseAccumulator> coop_c = coopmat<T, gl_ScopeSubgroup, COOP_K, COOP_K, gl_MatrixUseAccumulator>(0.0);

    // Compute row and col of the first element in the tile of the output
    // matrix handled by this subgroup
    uint tiles_M = M / COOP_M;
    uint n = (gsid / tiles_M) * COOP_N;
    uint m = (gsid % tiles_M) * COOP_M;

    for (uint k = 0; k < K; k += COOP_K){
        uint start_a = n * K + k;
        coopMatLoad(coop_a, data_a, start_a,  K,  gl_CooperativeMatrixLayoutRowMajor);
        
        #if TRANSPOSE_B
        uint start_b = m * K + k;
        coopMatLoad(coop_b, data_b, start_b,  K,  gl_CooperativeMatrixLayoutColumnMajor);
        #else
        uint start_b = k * M + m;
        coopMatLoad(coop_b, data_b, start_b,  M,  gl_CooperativeMatrixLayoutRowMajor);
        #endif

        coop_c = coopMatMulAdd(coop_a, coop_b, coop_c);
    }

    uint start_c = n * M + m;
    coopMatStore(coop_c, data_c, start_c, N, gl_CooperativeMatrixLayoutRowMajor);
    
}
