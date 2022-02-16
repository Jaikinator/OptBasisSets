from setuptools import setup

setup(
    name='optb',
    version='0.1.2.1',
    packages=['optb', 'optb/data'],
    install_requires = ["pyscf~=2.0.1",
                        "ase~=3.22.1",
                        "setuptools~=58.0.4",
                        "numpy~=1.21.2",
                        "pandas ~= 1.3.4",
                        "ase==3.22.1",
                        "basis-set-exchange==0.9",
                        "torch ~=1.10.0",
                        "tensorboard >= 2.7.0",
                        "dqc @ git+https://github.com/Jaikinator/dqc.git", # fork of dqc nightly
                        "xitorch @ git+https://github.com/Jaikinator/xitorch.git",   # fork of xitorch nightly
                        "knockknock>=0.1.8.1"], # fork of telegram message
    url='https://github.com/Jaikinator/OptBasisSets',
    license='',
    author='Jacob ',
    author_email='',
    description='setup pyscf and dqc to optimize gaussian basis '
)
