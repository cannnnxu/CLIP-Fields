from setuptools import setup, find_packages

setup(
    name="CLIP_Fields",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "sentence-transformers",
    ],
)