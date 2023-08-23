from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)
