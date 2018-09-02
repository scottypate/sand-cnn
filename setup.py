
# Instructions: https://packaging.python.org/tutorials/distributing-packages/

import setuptools

setuptools.setup(
    name="sand-cnn",
    url="https://github.com/scottypate/sand-cnn",
    author="Scotty Pate",
    description="Neural Net for image segmentation",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages('sand-cnn.sand_cnn')
)
