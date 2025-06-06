from setuptools import setup, find_packages

setup(
    name="rmvision",
    version="0.1.0",
    description="A package for RoboMaster vision processing",
    author="Yue Peng",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g. 'numpy', 'opencv-python'
    ],
    python_requires=">=3.6",
)
