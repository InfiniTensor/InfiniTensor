from random import random
import sys

def Gen_Tensor(size, filename):
    with open(filename, 'w') as fout:
        fout.write(' '.join([str(int(random()*10)) for i in range(size)]))
    fout.close()

now = sys.argv[1].split(',')
input_size = 1
for x in now:
    input_size *= int(x)
Gen_Tensor(input_size, 'input.dat')

now = sys.argv[2].split(',')
weight_size = 1
for x in now:
    weight_size *= int(x)
Gen_Tensor(weight_size, 'weight.dat')
