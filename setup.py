from setuptools import setup

setup(name='sketchedsgd',
      version='0.0.1',
      description='Sketched SGD',
      keywords='sketching sgd',
      url='https://github.com/dhroth/sketchedsgd.git',
      author='Daniel Rothchild',
      author_email='drothchild@berkeley.edu',
      license='GNU GPL-3.0',
      packages=['sketchedsgd'],
      install_requires=[
          'csvec',
      ],
      include_package_data=True,
      zip_safe=False)
