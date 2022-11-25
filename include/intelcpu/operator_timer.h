#pragma once
namespace infini {
namespace opTimer {
double getPerfConvMkl(int n, int c, int h, int w, int f, int r, int s, int padh,
                      int padw, int strideh, int stridew, int dilationh,
                      int dilationw, int group);

double getPerfConvTransposed2dMkl(int n, int c, int h, int w, int f, int r,
                                  int s, int padh, int padw, int strideh,
                                  int stridew, int dilationh, int dilationw,
                                  int oph, int opw, int group);

double getPerfMatmulMkl(int b, int m, int n, int k);
} // namespace opTimer
} // namespace infini
