from setuptools import setup
from setuptools import find_packages

setup(
    name='graphhop',
    version='1.0',
    description='GraphHop in Pytorch',
    author='Tian Xie',
    author_email='xietianfudan@gmail.com',
    url='https://tianxieusc.github.io',
    download_url='https://github.com/TianXieUSC/GraphHop',
    install_requires=[
        'numpy>=1.18.1',
        'torch>=1.5.0',
        'torchvision>=0.6.0',
        'torch-scatter>=1.2.0',
        'torch-sparse>=0.4.0',
        'torch-cluster>=1.3.0',
        'torch-geometric>=1.2.0',
        'networkx>=1.11',
        'scipy>=1.4.1'
    ],
    package_data={'graphhop': ['README.md']},
    packages=find_packages()
)
