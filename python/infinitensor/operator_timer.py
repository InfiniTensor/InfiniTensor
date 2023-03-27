from tokenize import Double
import pyinfinitensor  # import getPerfConv, getPerfMatmul


# FIXME: change API from getPerfOpDevice(...) to getPerfOp(device='dev', ...)
def getPerfConvCuda(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, group, name=""):
    return pyinfinitensor.getPerfConvCuda(n, c, h, w, f, r, s, padh, padw,
                                           strideh, stridew, dilationh, dilationw, group, name)


def getPerfConvTransposed2dCuda(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group):
    return pyinfinitensor.getPerfConvTransposed2dCuda(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group)


def getPerfMatmulCuda(b, m, n, k, name=""):
    return pyinfinitensor.getPerfMatmulCuda(b, m, n, k, name)


def getPerfConvMkl(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, group, name=""):
    return pyinfinitensor.getPerfConvMkl(n, c, h, w, f, r, s, padh, padw,
                                           strideh, stridew, dilationh, dilationw, group)


def getPerfConvTransposed2dMkl(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group):
    return pyinfinitensor.getPerfConvTransposed2dMkl(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group)


def getPerfMatmulMkl(b, m, n, k, name=""):
    return pyinfinitensor.getPerfMatmulMkl(b, m, n, k)
