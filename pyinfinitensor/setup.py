from setuptools import setup, find_packages

setup(
    name="pyinfinitensor",
    version="0.0.0",
    author="YdrMaster",
    author_email="ydrml@hotmail.com",
    description="Python frontend of InfiniTensor",
    url="",  # 你可以添加项目的URL，如果有的话
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"pyinfinitensor": ["*.so"]},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
    install_requires=[
        "onnx",
        "onnxsim",
    ],
    include_package_data=True,
)

