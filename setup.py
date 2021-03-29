import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rollout_tsp",
    version = "0.0.1",
    author = "Rodrigo Meneses",
    author_email = "rodrigo.menesesufc@gmail.com",
    description = ("Simples rollout package"),
    keywords = "example documentation",
    url = "https://github.com/rodrigofmeneses/tsp-rollout",
    packages=['rollout', 'tests'],
    # long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ],
)