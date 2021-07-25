#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs

from setuptools import find_packages, setup

DISTNAME = 'polire'
DESCRIPTION = 'A collection of interpolation methods.'
with codecs.open('README.md', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Zeel B Patel, Apoorv Agnihotri, S Deepak Narayanan'
MAINTAINER_EMAIL = 'patel_zeel@iitgn.ac.in, apoorv.agnihotri@iitgn.ac.in, deepak.narayanan@iitgn.ac.in'
URL = 'https://sustainability-lab.github.io/polire'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://sustainability-lab.github.io/polire'
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

INSTALL_REQUIRES = ['matplotlib', 'numpy', 'pandas', 'pykrige', 'scikit_learn',
                    'scipy', 'seaborn', 'Shapely', 'xgboost', 'GPy']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version='0.1.3',
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
