import setuptools

setuptools.setup(
    name="toc_azurewrapper",
    version="0.0.1",
    description=
        "Wrapper around Azure code, that simplifies model training and "
        "deployment for The Ocean Cleanup.",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
