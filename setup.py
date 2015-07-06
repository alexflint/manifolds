#!/usr/bin/env python
from distutils.core import setup

setup(name='manifolds',
      description='Utilities for working with differentiable manifolds',
      version='0.13',
      author='Alex Flint',
      author_email='alex.flint@gmail.com',
      url='https://github.com/alexflint/manifolds',
      packages=['manifolds'],
      package_dir={'manifolds': '.'},
      )
