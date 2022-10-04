from tokenize import Double
import pyinfinitensor  # import getPerfConv, getPerfMatmul


def getPerfConv(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, group, name=""):
    return pyinfinitensor.getPerfConvCudnn(n, c, h, w, f, r, s, padh, padw,
                                           strideh, stridew, dilationh, dilationw, group, name)


def getPerfConvTransposed2dCudnn(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group):
    return pyinfinitensor.getPerfConvTransposed2dCudnn(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group)


def getPerfMatmul(b, m, n, k, name=""):
    return pyinfinitensor.getPerfMatmulCublas(b, m, n, k, name)
