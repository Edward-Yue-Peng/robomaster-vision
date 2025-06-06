from setuptools import setup, find_packages

setup(
    name="rmvision",
    version="0.1.0",
    description="A package for RoboMaster vision processing",
    author="Yue Peng",
    packages=find_packages(),
    install_requires=[
        *[
            line.strip()
            for line in open("requirements.txt")
            if line.strip() and not line.startswith("#")
        ],
    ],
    python_requires=">=3.6",
)
