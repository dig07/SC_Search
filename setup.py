import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SC_search",
    version="0.0.1",
    author="Diganta Bandopadhyay and Christopher Moore",
    author_email="diganta@star.sr.bham.ac.uk",
    description="Semi-coherent search package for slowly chirping signals in LISA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dig07/SC_Search",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'matplotlib','scikit-learn','pathos','dill'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)