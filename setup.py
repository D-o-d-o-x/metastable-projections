from setuptools import setup, find_packages

setup(
    name='metastable-projections',
    version='1.0.0',
    # url='https://github.com/mypackage.git',
    # author='Author Name',
    # author_email='author@gmail.com',
    # description='Description of my package',
    packages=['.'],
    install_requires=['torch', 'stable_baselines3'],
)
