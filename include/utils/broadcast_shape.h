#pragma once

namespace infini {
void broadcastShape(const Shape &originShape, SmallArray &modifyShape,
                    int nDims, int size) {
    for (int i = nDims - 1; i >= 0; --i) {
        modifyShape.data[i] = 1;
    }
    for (int i = size - 1; i >= 0; --i) {
        modifyShape.data[i + nDims - size] = originShape[i];
    }
}

} // namespace infini
