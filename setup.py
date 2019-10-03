from setuptools import find_packages, setup

with open('requirements.txt', 'r') as req:
    install_requires = req.read().split("\n")

setup(
    name='recsys',
    packages=find_packages(),
    version='0.0.1',
    description='Recommendation Systems of Amazon Movies and TV',
    author='Jiawen Xu',
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
