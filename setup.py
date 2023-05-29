from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name = 'ml_project',
    version = '0.0.1',
    author = 'Abhiinv',
    author_email = 'abhiinv24@gmail.com',
    # checks which folders have __init__.py in them, each such folder behaves like a package
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)