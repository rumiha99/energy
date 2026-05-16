from setuptools import setup, find_packages

setup(
    name="size_module",
    version="1.0.0",
    author="oi-hiromu",
    packages=find_packages(),
    install_requires=["requests"],
    include_package_data=True,
)