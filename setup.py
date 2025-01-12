from setuptools import setup


setup(
    name='scratch_annotated',
    version='1.0.0',
    author='Soonwoo Kwon',
    author_email='soonoolimal@gmail.com',
    install_requires=['cupy==13.3.0',
                      'numpy==1.26.4'],
    description='Modules with annotations for 「Deep Learning from Scratch」 series.'
)