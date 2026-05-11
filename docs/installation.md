# Installation and setup
### Installation
**Install from PyPI:** 
The easiest way to install `its2s` is from PyPI.

`pip install its2s`

> **Prophet / CmdStan**: Prophet compiles a Stan model on first import. Expect ~5–10
> minutes the first time; subsequent imports are fast.

> **NeuralProphet / PyTorch**: NeuralProphet requires PyTorch (~1 GB) and is an optional
> dependency. Install it via `pip install its2s[neural]`.

> **setuptools pin**: The package pins `setuptools>=68,<71` because NeuralProphet uses
> `pkg_resources`, which was removed in setuptools 71+. If you see import errors related
> to `pkg_resources`, check your setuptools version.

You can also build `its2s` from source. Choose your environment manager:

=== "Conda"
    ```bash
    git clone https://github.com/causal-its/its2s.git
    cd its2s
    conda env create -f environment.yml
    conda activate its2s
    ```

    This creates a conda environment named `its2s` and installs the package in editable mode automatically.

=== "venv"
    ```bash
    git clone https://github.com/causal-its/its2s.git
    cd its2s
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    ```

    On Windows, replace `source venv/bin/activate` with `venv\Scripts\activate`.

*NOTE: We highly recommend you use the **same environment** whenever you run scripts. After setup, your working directory can be anywhere; imports work because `its2s` is installed into that environment.*
