from setuptools import find_packages, setup


setup(
    name="klomboagi",
    version="0.1.0",
    description="Experimental autonomous cognition runtime for persistent agent research",
    author="Ascendral",
    url="https://github.com/Ascendral/klomboagi",
    license="All rights reserved.",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    package_data={"klomboagi": ["config/*.json"]},
    entry_points={"console_scripts": ["klomboagi=klomboagi.interfaces.cli:main"]},
)
