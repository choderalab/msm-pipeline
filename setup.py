from setuptools import setup
import versioneer

setup(name='msmpipeline',
      version=versioneer.get_version(),
      description='Sensible defaults for MSMs',
      url='https://github.com/choderalab/msm-pipeline/',
      author='Sonya Hanson, Steven Albanese, Josh Fass',
      author_email='{sonya.hanson, steven.albanese, josh.fass}@choderalab.org',
      license='GNU Lesser General Public License v2 or later (LGPLv2+)',
      packages=['msmpipeline'],
      entry_points={
          'console_scripts': [
              'msm-pipeline = msmpipeline.pipeline:main',
              'test-report = msmpipeline.test_report_generation:main',
              'quick_test = msmpipeline.quick_test'
          ]
      },
      zip_safe=False)
