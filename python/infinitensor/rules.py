import pandas as pd
import numpy as np
from operator_timer import *
pd.options.display.float_format = '{:,.3f}'.format

df= pd.DataFrame(columns=['n', 'c', 'h', 'w', 'f', 'r', 's', 'ph', 'pw', 'sh', 'sw', 'dh', 'dw', 'oph', 'opw', 'group'])
def conv_original(name, n, c, h, w, f, r, s, ph, pw,
                            sh, sw, dh, dw, group):
    df.loc[name, ['n', 'c', 'h', 'w', 'f', 'r', 's', 'ph', 'pw', 'sh', 'sw', 'dh', 'dw', 'group']] = n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, group
    df.loc[name, 't_original'] = getPerfConv(n, c, h, w, f, r, s, ph, pw,
                        sh, sw, dh, dw, group)
    df.loc[name, 't_bias'] = getPerfConvBiasActCudnn(n, c, h, w, f, r, s, ph, pw,
                        sh, sw, dh, dw, group, bias=True)
    df.loc[name, 't_bias_relu'] = getPerfConvBiasActCudnn(n, c, h, w, f, r, s, ph, pw,
                        sh, sw, dh, dw, group, bias=True, act="Relu")

def conv_rule_5x5_to_3x3(name, n, c, h, w, f, r, s, ph, pw,
                            sh, sw, dh, dw, group):
    col = 't_5x5_to_3x3'
    if r == 5 and s == 5:
        df.loc[name, col] = getPerfConv(n, c, h, w, f*4, 3, 3, ph, pw,
                            sh, sw, dh, dw, group)
    else:
        df.loc[name, col] = np.inf

def conv_rule_9x9_to_3x3(name, n, c, h, w, f, r, s, ph, pw,
                            sh, sw, dh, dw, group):
    col = 't_9x9_to_3x3'
    if r == 9 and s == 9:
        df.loc[name, col] = getPerfConv(n, c, h, w, f*9, r//3, s//3, ph, pw,
                            sh, sw, dh, dw, group)
    else:
        df.loc[name, col] = np.inf

bandwidth=200*10**6 # (200GB/ms)

def conv_rule_conv2gemm(name, n, c, h, w, f, r, s, ph, pw,
                            sh, sw, dh, dw, group):
    col = 't_conv2gemm'
    if [sh, sw, dh, dw, group] == [1] * 5:
        # b = group
        # m = batch_size * input_height * input_width
        # n = output_channel * kernel_height * kernel_width
        # k = input_channel // group
        t_reduce= group*n*h*w*f*r*s*4/bandwidth if r>1 or s>1 else 0 
        df.loc[name, '_'+col+'_mem'] = t_reduce
        df.loc[name, col] = getPerfMatmul(group, n*h*w, f*r*s, c//group) + t_reduce 
    else:
        df.loc[name, col] = np.inf

# conv_rules=[conv_original, conv_rule_9x9_to_3x3, conv_rule_5x5_to_3x3, conv_rule_conv2gemm]
conv_rules=[conv_original]

def conv_tranpsposed2d_original(name, n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, oph, opw, group):
    df.loc[name, ['n', 'c', 'h', 'w', 'f', 'r', 's', 'ph', 'pw', 'sh', 'sw', 'dh', 'dw', 'oph', 'opw', 'group']] = n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, oph, opw, group
    df.loc[name, 't_original'] = getPerfConvTransposed2dCudnn(n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, oph, opw, group)

def conv_tranpsposed2d_togemm(name, n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, oph, opw, group):
    col = 't_conv2gemm'
    if [dh, dw, group] == [1] * 3:
        # ConvTransose2gemm
        # b = 1
        # m = batch_size * input_height*input_width
        # n = output_channel*kernel_height*kernel_width
        # k = input_channel
        t_reduce= n*h*w*c*r*s*4/bandwidth if r>1 or s>1 else 0 
        df.loc[name, '_'+col+'_mem'] = t_reduce
        print('t_conv2gemm', group, n*h*w, c*r*s, f)
        df.loc[name, col] = getPerfMatmul(group, n*h*w, c*r*s, f) + t_reduce 
    else:
        df.loc[name, col] = np.inf

conv_transposed2d_rules=[conv_tranpsposed2d_original, conv_tranpsposed2d_togemm]

def print_result():
    df['t_min'] = df.filter(regex=("^t_.*")).min(axis=1)
    print(df)
    print(f'Origin:  {df["t_original"].sum():.3f} ms')
    print(f'Min:     {df["t_min"].sum():.3f} ms') 
    print(f'Speedup: {df["t_original"].sum()/df["t_min"].sum():.3f} x')