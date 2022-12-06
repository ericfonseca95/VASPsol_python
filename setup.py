from setuptools import setup

setup(
    name='VASPsol',
    version='0.1.0',    
    description='A VASPsol python helper package to simplify calculations and analysis',
    author='Eric C. Fonseca',
    author_email='ericfonseca@ufl.edu',
    license='BSD 2-clause',
    packages=['VASPsol','custom'],
    install_requires=['dask',
                    'ase',
                    'pymatgen',
'seaborn',
'matplotlib',
'dask-jobqueue',
'dask-ml',
'xlrd'
                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
