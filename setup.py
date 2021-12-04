from setuptools import setup

setup(
    name='OptBasisSets',
    version='0.1',
    packages=['optb'],
    install_requires = ["dqc~=0.2.0.dev1172031026566",
                        "xitorch~=0.3.0",
                        "pyscf~=2.0.1",
                        "ase~=3.22.1",
                        "setuptools~=58.0.4",
                        "numpy~=1.21.2",
                        "ase==3.22.1",
                        "basis-set-exchange==0.9",
                        "torch==1.10.0"],
    url='https://github.com/Jaikinator/OptBasisSets',
    dependency_links=["https://github.com/Jaikinator/dqc/tarball/master#egg=dqc.0.2.0.dev1172031026566"],
    license='',
    author='Jacob ',
    author_email='',
    description='setup pyscf and dqc to optimize gaussian basis '
)
