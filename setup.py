#!/usr/bin/env python
''' package setup/installation and metadata for lambdata ''' 

import setuptools

REQUIRED = [
        'numpy', 
        'pandas'
        ]

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
        name="lambdata-quinndougherty", 
        version="0.0.2", 
        author="quinndougherty", 
        description="A collection of data science helper functions", 
        long_description=LONG_DESCRIPTION, 
        long_description_content_type="text/markdown",
        url = "https://github.com/quinn-dougherty/lambdata",
        packages=setuptools.find_packages(),
        python_requires=">=3.5",
        install_requires=REQUIRED,
        classifiers=[
		"Programming Language :: Python :: 3",
        	"Operating System :: OS Independent"
		]
        )
