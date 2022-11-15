import os


def eval(filename, kernel, shape):
    with open("../eval_pfusion/eval_kernel.tmp", "r") as f:
        code = f.read()
    code = code.replace("%%invoke_func%%", kernel)
    code = code.replace("%%shape%%", shape)
    with open("../generated_code/tmp.cu", "w") as f:
        f.write(code)
    # os.system("make -j && ./test_bias")
    os.system(
        "nvcc ../generated_code/tmp.cu ../generated_code/" + filename + " -I ../eval_pfusion -o ./tmp")
    os.system("./tmp")


if __name__ == "__main__":
    eval("bias_0.cu", "invoke_func_0", "{28 * 28, 24}")
    eval("bias_1.cu", "invoke_func_1", "{28 * 28, 58}")
    eval("bias_2.cu", "invoke_func_2", "{14 * 14, 116}")
    eval("bias_3.cu", "invoke_func_3", "{7 * 7, 232}")
