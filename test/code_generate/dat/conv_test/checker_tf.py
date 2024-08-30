import tensorflow as tf
import argparse
import numpy as np

input_size  = (1, 1, 1, 1)
weight_size = (1, 1, 1, 1)

stride   = 1 # default
padding  = 0 # default
dilation = 1 # default
groups   = 1 # default

input_filename  = 'input.dat'
weight_filename = 'weight.dat'
output_filename = 'output.dat'
output_std_filename = 'output_tpm.dat'

EPS = 1e-5

def Init_Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_size', type=str, default='1,1,1,1')
    parser.add_argument('-w', '--weight_size', type=str, default='1,1,1,1')
    parser.add_argument('-s', '--stride', type=str, default='1')
    parser.add_argument('-p', '--padding', type=str, default='0')
    parser.add_argument('-d', '--dilation', type=str, default='1')
    parser.add_argument('-g', '--groups', type=int, default=1)
    return parser.parse_args()

def Str2Tuple(st):
    st = st.replace(' ', '').split(',')
    return tuple([int(x) for x in st])

def IntStr2Tuple(st):
    if not ',' in st:
        return int(st)
    return Str2Tuple(st)

def Get_Args(args):
    global input_size, weight_size, stride, padding, dilation, groups
    input_size = Str2Tuple(args.input_size)
    weight_size = Str2Tuple(args.weight_size)
    stride = IntStr2Tuple(args.stride)
    padding = IntStr2Tuple(args.padding)
    dilation = IntStr2Tuple(args.dilation)
    groups = args.groups

# nchw oihw
def Input_Tensor(filename, size, change=False):
    with open(filename, 'r') as fin:
        inLines = fin.readlines()
    fin.close()

    rtn = []
    for l in inLines:
        now = l.strip().split(' ')
        rtn = rtn + [float(x) for x in now]
    rtn = np.array(rtn).reshape(size)
    if not change:
        return rtn
    now = rtn.tolist()
    rtn = np.zeros(size[2], size[3], size[1], size[0]).tolist()
    for o in range(size[0]):
        for i in range(size[1]):
            for h in range(size[2]):
                for w in range(size[3]):
                    rtn[h][w][i][o] = now[o][i][h][w]
    return np.array(rtn)

def TF_Change(input_tensor, # padding
              weight_tensor, # oihw -> hwio
              stride, # tuple -> list
              padding,
              dilation): # tuple -> list
    now = input_tensor.tolist()
    rtn = []
    if type(padding) == tuple:
        pdh, pdw = padding[0], padding[1]
    else:
        pdh, pdw = padding, padding
    N, C, H, W = len(now), len(now[0]), len(now[0][0]), len(now[0][0][0])
    nh, nw = H + pdh*2, W + pdw*2
    for n in range(N):
        rtn.append([])
        for c in range(C):
            rtn[n].append([])
            for h in range(nh):
                rtn[n][c].append([])
                for w in range(nw):
                    if h >= pdh and w >= pdw and h < H+pdh and w < W+pdw:
                        rtn[n][c][h].append(now[n][c][h-pdh][w-pdw])
                    else:
                        rtn[n][c][h].append(0.0)
    input_tensor = np.array(rtn)

    now = weight_tensor.tolist()
    rtn = []
    O, I, H, W = len(now), len(now[0]), len(now[0][0]), len(now[0][0][0])
    for h in range(H):
        rtn.append([])
        for w in range(W):
            rtn[h].append([])
            for i in range(I):
                rtn[h][w].append([])
                for o in range(O):
                    rtn[h][w][i].append(now[o][i][h][w])
    weight_tensor = np.array(rtn)

    if type(stride) == tuple:
        stride = [1, 1, stride[0], stride[1]]
    if type(dilation) == tuple:
        dilation = [1, 1, dilation[0], dilation[1]]

    return input_tensor, weight_tensor, stride, dilation

def Output_Tensor(filename, opt):
    opt = opt.numpy().reshape(-1).tolist()
    with open(filename, 'w') as fout:
        fout.write(' '.join([str(x) for x in opt]))

def Compare(output_tensor):
    with open(output_std_filename, 'r') as fin:
        inLines = fin.readlines()
    fin.close()

    rtn = []
    for l in inLines:
        now = l.strip().split(' ')
        rtn = rtn + [float(x) for x in now]
    std = rtn
    ans = output_tensor.reshape(-1).tolist()
    diff = 0
    if len(std) != len(ans):
        print('Unequal Length!')
        return
    for i in range(len(ans)):
        a, b = std[i], ans[i]
        diff = max(diff, abs(a-b) / b)
    if diff < EPS:
        print('Right Output!!! diff = %f' % diff)
    else:
        print('Wrong Answer!!! diff = %f' % diff)

def Check_Conv2d():
    input_tensor = Input_Tensor(input_filename, input_size)
    weight_tensor = Input_Tensor(weight_filename, weight_size)

    global stride, padding, dilation
    input_tensor, weight_tensor, stride, dilation = TF_Change(input_tensor, weight_tensor, stride, padding, dilation)
    
    output_tensor = tf.nn.conv2d(input=input_tensor, 
                                 filters=weight_tensor, 
                                 strides=stride, 
                                 padding='VALID', 
                                 data_format='NCHW',
                                 dilations=dilation)

    Output_Tensor(output_filename, output_tensor)
    Compare(np.array(output_tensor))

def main():
    Get_Args(Init_Args())    
    Check_Conv2d()

if __name__ == '__main__':
    main()
