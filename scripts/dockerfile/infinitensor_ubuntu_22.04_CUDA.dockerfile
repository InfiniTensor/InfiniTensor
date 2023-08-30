FROM nvcr.io/nvidia/pytorch:23.07-py3

# Install dependencies.
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install -y git make cmake build-essential python-is-python3 python-dev-is-python3 python3-pip libdw-dev openssh-client
# Generate ssh key.
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -P ""

# Update pip and switch to Tsinghua source.
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# TODO: Since SSH clone repo requires adding the SSH key to the GitHub profile, 
# the process of pulling the project and compiling it has been temporarily commented.

# Download InfiniTensor.
# RUN git clone git@github.com:InfiniTensor/InfiniTensor.git /root?/InfiniTensor --branch master --single-branch --recursive

# Build and Install InfiniTensor
# RUN cd /root/InfiniTensor && make install-python CUDA=ON
