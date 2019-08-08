# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from bs4 import BeautifulSoup as soup
from markdown import markdown
description = markdown(open('README.md').read())
#text = ''.join(soup(description, 'lxml').findAll(text=True))
setup(name="AutoAiLib",
      version ='1.0.1',
      packages=find_packages(),
      description='The library that automates the silly ML things.',
      license='GNU GPLv3',
      long_description=description,
      long_description_content_type = 'text/markdown',
      author ='Matthew Mulhall',
      author_email='matthewlmulhall@gmail.com',
      install_requires=['scikit-image',  'progressbar2', 'pandas', 'keras', 'numpy', 'opencv-python','scipy',],
      classifiers= ['Programming Language :: Python :: 3']
      )