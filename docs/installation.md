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

You can also build `its2s` from source with:

```bash
git clone https://github.com/causal-its/its2s.git
cd its2s
python -m pip install -e .
```

### Environment setup
**Conda environment:** If you use conda, you can create the `its2s` environment and install `its2s` in editable mode with the following commands. This will create an environment called `its2s` and install `its2s` in editable mode. From the **repository root** (where `pyproject.toml` and `environment.yml` live):
```bash
conda env create -f environment.yml
conda activate its2s
```

If you don’t use conda, create a virtual environment (or use your usual setup) and run `pip install -e .` from the repo root—the same command works for any active Python you intend to use. Use `pip install .` instead if you don’t want an editable install.


*NOTE: We highly recommend you use the **same environment** whenever you run scripts. After setup, your working directory can be anywhere; imports work because `its2s` is installed into that environment.*
