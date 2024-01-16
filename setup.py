from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="openloss",
    version="0.0.1.1",
    description="Cost functions designed for open-set classification tasks, namely, Entropic Open-set, ObjectoSphere and Maximal-Entropy Loss.",
    packages=find_packages(where="python"),
    package_dir={"" : "python"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaelvareto/maximal-entropy-loss",
    author="Rafael Vareto",
    author_email="rafael@vareto.com.br",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["torch", "pytest>=7.0"],
    python_requires=">=3.8",
)
