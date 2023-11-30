#pragma once

namespace infini {
void broadcastShape(const Shape &originShape, SmallArray &modifyShape,
                    int nDims, int size) {
    for (int i = nDims - size - 1; i >= 0; --i) {
        modifyShape.data[i] = 1;
    }
    for (int i = nDims - 1; i >= nDims - size; --i) {
        modifyShape.data[i] = originShape[i - nDims + size];
    }
}

} // namespace infini
