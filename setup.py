import re
import ast
import os
from setuptools import setup, find_packages

def get_version(init_file):
    _version_re = re.compile(r'__version__\s+=\s+(.*)')
    with open(init_file, 'rb') as f:
        version = str(ast.literal_eval(_version_re.search(f.read().decode('utf-8')).group(1)))
    return version

package_name = "datatools"
author_url = 'https://github.com/droyed'

version = get_version(package_name+"/__init__.py")
repo_url = os.path.join(author_url, package_name)

setup(name=package_name,
      version=version,
      description='Data preparation tools for Python',
      url=repo_url,
      author='Divakar Roy',
      author_email='droygatech@gmail.com',
      license='MIT',
      classifiers=[
      'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Operating System :: POSIX :: Linux',
          ],
      keywords='artificial intelligence deep learning tensorflow data image object detection tools', 
      packages=find_packages(),
      package_data={ "": ["../datatools/od_template.xml"]},
      include_package_data=True,
      zip_safe=False)
