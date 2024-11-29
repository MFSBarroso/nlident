from setuptools import setup, find_packages

setup(
    name='nlindent',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    author='Márcio Falcão Santos Barroso e Jim Jones da Silveira Marciano',
    author_email='barroso@ufsj.edu.br',
    description='nlindent Package for Nonlinear Systems Identification',
    url='http://github.com/nlindentpy/nlindent',
)
