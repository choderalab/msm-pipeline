from setuptools import setup
import versioneer

setup(name='msmpipeline',
      version=versioneer.get_version(),
      description='Sensible defaults for MSMs',
      url='http://github.com/storborg/funniest',
      author='Sonya Hanson, Steven Albanese, Josh Fass',
      author_email='{sonya.hanson, steven.albanese, josh.fass}@choderalab.org',
      license='MIT',
      packages=['msmpipeline'],
      zip_safe=False)
