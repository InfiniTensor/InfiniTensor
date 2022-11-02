#pragma once
#include "core/operator.h"
namespace infini {
namespace opTimer {

double getPerfConvCudnn(int n, int c, int h, int w, int f, int r, int s,
                        int padh, int padw, int strideh, int stridew,
                        int dilationh, int dilationw, int group);

double getPerfConvBiasActCudnn(int n, int c, int h, int w, int f, int r, int s,
                               int padh, int padw, int strideh, int stridew,
                               int dilationh, int dilationw, int group,
                               bool bias, string act);

double getPerfConvTransposed2dCudnn(int n, int c, int h, int w, int f, int r,
                                    int s, int padh, int padw, int strideh,
                                    int stridew, int dilationh, int dilationw,
                                    int oph, int opw, int group);

double getPerfMatmulCublas(int b, int m, int n, int k, const char *name);
} // namespace opTimer
} // namespace infini
