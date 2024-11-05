# Python

## UV

An extremely fast Python package and project manager, written in Rust.

[Github](https://github.com/astral-sh/uv)

### Install
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Manage Python
```sh
# show installed version
uv python list

# install python version
uv python install 3.11.9
```

### Manage Project and Env
```sh
# Init project and set  minimum supported Python version
uv init --python 3.9

# Create virtual env with specific python version
uv venv --python 3.9.16

# Activate virtual env
source .venv/bin/activate
```

### Install Packages
```sh
uv add package
```

### Run Python
```sh
uv run script.py
```
