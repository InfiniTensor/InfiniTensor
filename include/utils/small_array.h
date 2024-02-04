#pragma once
namespace infini {

#define SMALL_ARRAY_SIZE 8
struct SmallArray {
    int data[SMALL_ARRAY_SIZE];
    
    int prod(int start, int end) {
        int result = 1;
        for(int i = start; i < end; ++i){
            result *= data[i];
        }
        return result;
    }
};

} // namespace infini
