from setuptools import setup, find_packages

setup(
    name='mlutils',
    version='0.0.4',
    author='Adrian Wälchli',
    author_email='adrian.waelchli@inf.unibe.ch',
    scripts=[],
    url='https://github.com/awaelchli/mlutils.git',
    license='LICENSE.txt',
    description='Machine Learning Utilities',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'torch>=1.2.0',
        'torchvision>=0.4.0'
        'numpy>=1.18',
        'matplotlib>=3.1',
        'Pillow==6.1',
        'pytest',
    ],
)
