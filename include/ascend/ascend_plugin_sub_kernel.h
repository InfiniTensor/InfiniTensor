#pragma once
// #include "core/data_type.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using Shape = std::vector<int>;
struct PluginMetaData {
    Shape input_shape;
    Shape output_shape;
    size_t input_size;
    size_t output_size;
    // DataType dataType;
    int kernel_size;
    int stride;
};

void plugin_sub_kernel(float *in, float *out, PluginMetaData metaData,
                       void *stream);