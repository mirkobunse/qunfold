from setuptools import setup

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="qunfold",
    version="0.1.1",
    description="Composable quantification and unfolding methods",
    long_description=readme(),
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
    ],
    keywords=[
        "machine-learning",
        "supervised learning",
        "quantification",
        "supervised prevalence estimation",
        "unfolding",
    ],
    url="https://github.com/mirkobunse/qunfold",
    author="Mirko Bunse",
    author_email="mirko.bunse@tu-dortmund.de",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    extras_require = {
        "experiments" : ["jax[cpu]", "pandas", "quapy", "scikit-learn", "tqdm"],
        "tests" : ["jax[cpu]", "nose", "quapy", "scikit-learn"],
        "docs" : ["jax[cpu]", "myst-parser", "quapy", "scikit-learn", "sphinx-rtd-theme"],
    }
)
