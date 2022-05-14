from setuptools import setup

from setuptools import setup

setup(
    name="Penrisk",
    version="0.0.1",
    packages=["penrisk"],
    entry_points={
        "console_scripts": ["tiler=penrisk.cli:tiler"],
    },
    install_requires=["numpy"],
)
