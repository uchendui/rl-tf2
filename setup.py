from setuptools import setup, find_packages

setup(name='ike_rllib', version='2.0', packages=find_packages(),
      install_requires=['tensorflow', 'numpy', 'gym', 'absl'])
