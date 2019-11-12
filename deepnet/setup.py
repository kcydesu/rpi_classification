from setuptools import setup

setup(
    name='deepnet',
    version='1.0.0',
    packages=['deepnet'],
    url='',
    license='',
    author='Casey Bruno',
    author_email='cabruno98@gmail.com',
    description='Class using AlexNet to classify images',
    install_requires=['torch','torch-vision'],
    package_data={'data', ['*.txt']}
)
