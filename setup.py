from setuptools import setup
from setuptools import find_packages

setup(
    name='wtte-torch',
    version='0.1',
    description='Weibull Time To Event prediction with PyTorch and deep learning.',
    author='Aaron Epel',
    author_email='aaron.epel@gmail.com',
    license='MIT',
    install_requires=[
        'pytorch>=1.5',
        'numpy>=1.18',
        'pandas>=1.0',
        'scipy>=1.5',
        'six==1.10.0',
    ],
    packages=find_packages('.', exclude=['examples', 'tests']),
)