'''google cloud ml-engine package config
'''
from setuptools import setup, find_packages

setup(name='hands_test',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='keras cnn',
      author='frederik laboyrie',
      author_email='frederiklaboyrie1@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'],
      zip_safe=False)