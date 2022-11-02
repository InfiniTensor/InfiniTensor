import argparse
from tokenize import Double
import pyinfinitensor  # import getPerfConv, getPerfMatmul


def getPerfConv(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, group):
    return pyinfinitensor.getPerfConvCudnn(n, c, h, w, f, r, s, padh, padw,
                                           strideh, stridew, dilationh, dilationw, group)


def getPerfConvTransposed2dCudnn(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group):
    return pyinfinitensor.getPerfConvTransposed2dCudnn(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, oph, opw, group)


def getPerfMatmul(b, m, n, k, name=""):
    return pyinfinitensor.getPerfMatmulCublas(b, m, n, k, name)


def getPerfConvBiasActCudnn(n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, group, bias: bool, act="None"):
    return pyinfinitensor.getPerfConvBiasActCudnn(n, c, h, w, f, r, s, padh, padw,
                                                  strideh, stridew, dilationh, dilationw, group, bias, act)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('op', metavar='operator', type=str)
    parser.add_argument('shape', nargs='+')
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--act', type=str, default="None")
    args = parser.parse_args()
    print(args)
    if args.op == 'gemm':
        t = getPerfMatmul(int(args.shape[0]), int(
            args.shape[1]), int(args.shape[2]), int(args.shape[3]))
        print(
            f'time {t:.3f} ms, {2*int(args.shape[0])*int(args.shape[1])*int(args.shape[2])*int(args.shape[3])/t/1e9:.3f} TFLOPS')
    elif args.op == 'conv':
        assert len(args.shape) == 7
        n, c, h, w, f, r, s = [int(v) for v in args.shape]
        padh = padw = int(args.pad)
        strideh = stridew = int(args.stride)
        dilationh = dilationw = int(args.dilation)
        group = int(args.group)
        bias = int(args.bias)
        act = args.act
        assert group==1, "Unsupported"
        t = pyinfinitensor.getPerfConvBiasActCudnn(
            n, c, h, w, f, r, s, padh, padw, strideh, stridew, dilationh, dilationw, group, bias, act)
        print(
            f'time {t:.3f} ms, {n*c*h*w*f*r*s/strideh/stridew*2/10**9:.3f} TFlops')
    else:
        assert False, "Not supported"
