from setuptools import setup, find_packages

# install_requires = open_requirements('requirements.txt')

# d = {}
# exec(open("spikefinder/version.py").read(), None, d)
# version = d['version']

long_description = open("README.md").read()

setup(name='ensongdec',
      version='1.0',
      description='Python toolkit to build EnCodec-based vocoders.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/pabloslash/enSongDec",
      packages=find_packages(),
      author='Pablo M. Tostado',
      author_email='tostadomarcos@gmail.com',
      license='MIT',
      install_requires=['numpy',
                        'matplotlib',
                        'pandas',
                        'librosa'
                       ],
      
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
      zip_safe=False)
