import os


def eval(filename, kernel, shape, perm):
    with open("../eval_pfusion/eval_transpose.tmp", "r") as f:
        code = f.read()
    code = code.replace("%%invoke_func%%", kernel)
    code = code.replace("%%shape%%", shape)
    code = code.replace("%%perm%%", perm)
    with open("../generated_code/tmp.cu", "w") as f:
        f.write(code)
    # os.system("make -j && ./test_bias")
    os.system(
        "nvcc ../generated_code/tmp.cu ../generated_code/" + filename + " -I ../eval_pfusion -o ./tmp")
    os.system("./tmp")


if __name__ == "__main__":
    eval("transpose_0.cu", "invoke_func_0", "{28 * 28, 58, 2}", "{0, 2, 1}")
    eval("transpose_1.cu", "invoke_func_1", "{14 * 14, 116, 2}", "{0, 2, 1}")
    eval("transpose_2.cu", "invoke_func_2", "{7 * 7, 232, 2}", "{0, 2, 1}")
