
from setuptools import setup, find_packages
from pathlib import Path

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'gp_grief' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)

setup(
    name='gp_grief',
    author=about['__author__'],
    #author_email=about['__author_email__'],
    #license=about['__license__'],
    version=about['__version__'],
    url='https://github.com/scwolof/gp_grief',
    packages=find_packages(exclude=['tests','docs']),
    install_requires=['numpy>=1.7',
                      'scipy>=0.17',
                      'matplotlib>=1.3',
                      'six',
                      'paramz>=0.9.0',
                      'GPy'],
    setup_requires=['numpy>=1.7',
                    'scipy>=0.17',
                    'matplotlib>=1.3',
                    'six',
                    'paramz>=0.9.0',
                    'GPy'],
    tests_require=['pytest-runner','pytest', 'pytest-cov'],
)
