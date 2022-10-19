project(PowerFusion)

aux_source_directory(. SRC_CXX)
aux_source_directory(micro_kernel/. SRC_CXX)
cuda_add_library(pf_base SHARED ${SRC_CXX})
set_target_properties(pf_base PROPERTIES CUDA_ARCHITECTURES "80")
