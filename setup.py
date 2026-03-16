from setuptools import setup, find_packages

setup(
    name="astats-nl-layer",
    version="0.1.0",
    author="Atta ul Asad",
    author_email="fa22-bcs-073@cuivehari.edu.pk",
    description="Natural Language Understanding layer for AStats — GSoC 2026 INCF Project #33",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/attaulasad/astats-nl-layer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.38.0",
        "torch>=2.0.0",
        "rich>=13.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
