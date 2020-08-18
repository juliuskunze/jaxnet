import pathlib

import setuptools

setuptools.setup(
    name='jaxnet',
    version='0.2.7',
    description='Neural Nets for JAX',
    url='http://github.com/JuliusKunze/jaxnet',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    install_requires=['jax>=0.1.42', 'dill'],
    packages=['jaxnet'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research'
    ]
)
