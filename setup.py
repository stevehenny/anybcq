from setuptools import find_packages, setup

install_requires = [
    'torch',
    'numpy',
    'transformers==4.51.0',
    'datasets',
    'evaluate',
    'lm_eval==0.4.5',
    'peft==0.10.0'
]


setup(
    name='anybcq',
    version='0.0',
    author='Anonymous',
    author_email='Anonymous',
    description='Repository for anybcq research',
    packages=find_packages(),
    install_requires=install_requires,
)
