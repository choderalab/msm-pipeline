from setuptools import setup

setup(name='msmpipeline',
      version='0.0.1',
      description='Sensible defaults for MSMs',
      url='https://github.com/choderalab/msm-pipeline/',
      author='Sonya Hanson, Steven Albanese, Josh Fass',
      author_email='{sonya.hanson, steven.albanese, josh.fass}@choderalab.org',
      license='GNU Lesser General Public License v2 or later (LGPLv2+)',
      packages=['msmpipeline'],
      entry_points={
          'console_scripts': [
              'msm-pipeline = msmpipeline:main',
              'align-pdbs = msmpipeline.align_pdbs:main'
          ]
      },
      zip_safe=False)