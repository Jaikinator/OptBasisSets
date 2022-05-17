import re
import os
from setuptools import setup, find_packages

module_name = "optb"
github_url = "https://github.com/Jaikinator/OptBasisSets.git"

# open readme and convert all relative path to absolute path
#descriptioninf for the module
# with open("README.md", "r") as f:
#     long_desc = f.read()
#
# link_pattern = re.compile(r"\(([\w\-/]+)\)")
#
# link_repl = r"(%s\1)" % github_url
# long_desc = re.sub(link_pattern, link_repl, long_desc)


file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "version.py"))
version = {"__file__": verfile}

with open(verfile, "r") as fp:
    exec(fp.read(), version)

############### setup ###############

build_version = "OPTB_BUILD" in os.environ

setup(
    name=module_name,
    version=version["get_version"](build_version),
    packages=find_packages(),
    python_requires="~=3.9",
    install_requires = ["pyscf~=2.0.1",
                        "ase~=3.22.1",
                        "setuptools~=58.0.4",
                        "numpy >= 1.21.2",
                        "pandas >= 1.3.4",
                        "ase==3.22.1",
                        "basis-set-exchange==0.9",
                        "torch ~=1.10.0",
                        "tensorboard >= 2.7.0",
                        "dqc @ git+https://github.com/Jaikinator/dqc.git", # fork of dqc nightly
                        "xitorch @ git+https://github.com/Jaikinator/xitorch.git",   # fork of xitorch nightly
                        "knockknock>=0.1.8.1"], # fork of telegram message
    url='https://github.com/Jaikinator/OptBasisSets',
    license='',
    author='Jaikinator',
    author_email='',
    description='optimize gaussian basis sets'
)
