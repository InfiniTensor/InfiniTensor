import tensorflow as tf
import argparse
import numpy as np

input_size  = (1, 1, 1, 1)

input_filename  = 'input.dat'
output_filename = 'output.dat'
output_std_filename = 'output_tpm.dat'

EPS = 1e-5

def Init_Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_size', type=str, default='1,1,1,1')
    return parser.parse_args()

def Str2Tuple(st):
    st = st.replace(' ', '').split(',')
    return tuple([int(x) for x in st])

def IntStr2Tuple(st):
    if not ',' in st:
        return int(st)
    return Str2Tuple(st)

def Get_Args(args):
    global input_size
    input_size = Str2Tuple(args.input_size)

# nchw
def Input_Tensor(filename): # -> List
    with open(filename, 'r') as fin:
        inLines = fin.readlines()
    fin.close()

    rtn = []
    for l in inLines:
        now = l.strip().split(' ')
        rtn = rtn + [float(x) for x in now]
    return rtn

def Output_Tensor(filename, opt):
    opt = opt.numpy().reshape(-1).tolist()
    with open(filename, 'w') as fout:
        fout.write(' '.join([str(x) for x in opt]))
        fout.close()

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
        diff = max(diff, abs(a-b) / (abs(b)+0.01))
    if diff < EPS:
        print('Right Output!!! diff = %f' % diff)
    else:
        print('Wrong Answer!!! diff = %f' % diff)

def Check_Transpose():
    input_tensor = Input_Tensor(input_filename)

    output_tensor = input_tensor
    with open('transpose_ops.op', 'r') as fin:
        inLines = fin.readlines()
        fin.close()
    for line in inLines:
        now = line.strip().split()
        if not now:
            continue
        if now[0] == 'reshape':
            shape = [int(x) for x in now[1:]]
            output_tensor = tf.reshape(output_tensor, shape)
        if now[0] == 'transpose':
            perm = [int(x) for x in now[1:]]
            output_tensor = tf.transpose(output_tensor, perm=perm)

    Output_Tensor(output_filename, output_tensor)
    Compare(np.array(output_tensor))

def main():
    Get_Args(Init_Args())    
    Check_Transpose()

if __name__ == '__main__':
    main()
