from setuptools import setup


def read(fname):
    import os
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='keras-layernorm-rnn',
      version='0.1.1',
      description='RNNs with layer normalization',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/kmedian/keras-layernorm-rnn',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['keras_layernorm_rnn'],
      install_requires=[
          'setuptools>=40.0.0',
          'tensorflow==2.1.0',
          'numpy>=1.18.0'],
      python_requires='>=3.7.5',
      zip_safe=False)
