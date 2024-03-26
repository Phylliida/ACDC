import setuptools

setuptools.setup(
    name = "rnn_acdc",
    version = "0.0.1",
    author = "Phylliida",
    author_email = "phylliidadev@gmail.com",
    description = "ACDC (Automated Circuit Discovery) port for Mamba",
    url = "https://github.com/Phylliida/RNN-ACDC.git",
    project_urls = {
        "Bug Tracker": "https://github.com/Phylliida/RNN-ACDC/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = ">=3.6",
    install_requires = ['transformer-lens', 'torch', 'einops', 'jaxtyping', 'mamba_lens @ git+https://github.com/Phylliida/MambaLens.git']
)
