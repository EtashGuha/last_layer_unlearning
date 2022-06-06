from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="testbed",
    install_requires=required,
    #package_dir={"": "src"},
    #packages=find_packages("src"),
    packages=["testbed"],
    version="1.0",
)
