from setuptools import setup, find_packages

# Get the documentation
with open("README.md", "r") as fh:
    long_description = fh.read()

# Get the requirements
with open('requirements_frozen.txt') as f: required_frozen = f.read().splitlines()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.9",
]

setup(
    name="hpvsim_methods_manuscript",
    version="1.0.0",
    author="Robyn Stuart, Jamie Cohen, Cliff Kerr, Romesh Abeysuriya, Mariah Boudreau, and Dan Klein",
    author_email="info@hpvsim.org",
    description="Code for the 'HPVsim: An agent-based model of HPV transmission and cervical disease' paper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://hpvsim.org',
    keywords=["COVID-19", "SARS-CoV-2", "testing", "tracing", "quarantine", "covasim"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "sciris",
        "hpvsim",
    ],
    extras_require={
        "frozen": required_frozen,
    }
)
