from setuptools import setup
import versioneer

setup(name='msmpipeline',
      version=versioneer.get_version(),
      description='Sensible defaults for MSMs',
      url='https://github.com/choderalab/msm-pipeline/',
      author='Sonya Hanson, Steven Albanese, Josh Fass',
      author_email='{sonya.hanson, steven.albanese, josh.fass}@choderalab.org',
      license='MIT',
      packages=['msmpipeline'],
      entry_points={
          'console_scripts': [
              'msm-pipeline = msmpipeline.pipeline:main',
              'test-report = msmpipeline.test_report_generation:main'
          ]
      },
      zip_safe=False)
