# Installing OpenEuroLLM oellm-autoexp on LUMI HPC

This guide outlines the procedure to install the `oellm-autoexp` tool on LUMI. Because LUMI environments often have strict quotas on the `$HOME` directory and SSH keys may not be configured for submodules, this guide uses manual workarounds to route all installations and caches to a designated scratch directory.

## Prerequisites: Define Workspace Paths
Set up your active project scratch space. In this guide, we use `project_462001516` as the active project.

```bash
# Define your active scratch directory
export SCRATCH_DIR="/flash/project_462001516/users/$USER"

# Navigate to your workspace
cd $SCRATCH_DIR
```

## Step 1: Clone the Repository (HTTPS Workaround)
The repository uses SSH links for its submodules. If you do not have SSH keys configured on LUMI, the submodule cloning will fail. Force Git to use HTTPS instead:

```bash
# Force Git to rewrite SSH URLs to HTTPS globally
git config --global url."https://github.com/".insteadOf git@github.com:

# Clone the repository and its submodules
git clone https://github.com/OpenEuroLLM/oellm-autoexp.git --recurse-submodules
cd oellm-autoexp
```
*(Optional: You can revert the git config later with `git config --global --unset url."https://github.com/".insteadOf`)*

## Step 2: Install `uv` Manually
The default `uv` install script may attempt to write to inaccessible legacy project directories linked to your user profile. To bypass this, download and place the binaries manually in your active scratch space:

```bash
# Define your local bin directory in the active scratch space
export INSTALL_DIR="$SCRATCH_DIR/.local/bin"
mkdir -p $INSTALL_DIR

# Download the pre-compiled uv binary (v0.4.15)
wget -qO- https://github.com/astral-sh/uv/releases/download/0.4.15/uv-x86_64-unknown-linux-gnu.tar.gz | tar xvz

# Move the executables to your installation directory
mv uv-x86_64-unknown-linux-gnu/uv $INSTALL_DIR/
mv uv-x86_64-unknown-linux-gnu/uvx $INSTALL_DIR/

# Clean up the extracted archive folder
rm -rf uv-x86_64-unknown-linux-gnu

# Add the directory to your PATH
export PATH="$INSTALL_DIR:$PATH"

# Verify the installation
uv --version
```

## Step 3: Redirect `uv` Caches
To prevent `uv` from filling up your limited `$HOME` storage or attempting to write to restricted folders, redirect its cache and python installation paths to your scratch space:

```bash
export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/.local/share/uv/python"

# Create the cache directories
mkdir -p $UV_CACHE_DIR $UV_PYTHON_INSTALL_DIR
```

## Step 4: Create the Environment
With `uv` installed and caching properly redirected, you can now sync the project. This will download dependencies and create the `.venv` directory inside the project folder.

```bash
uv sync
```

## Step 5: Sanity Check
Run the provided speed test to confirm the environment is working correctly and interacting with Slurm. The job should finish in ~5 minutes.

```bash
SLURM_ACCOUNT=project_462001516 SLURM_PARTITION=dev-g uv run python scripts/run_autoexp.py --config-name experiments/megatron_lumi_speed_test
```

# Running of DataMix experiments

There is one config file per model size, `scale/*.*B.yaml`.
To run training, conversion and evaluation of all sizes in one go, see [chain_guide.md](chain_guide.md).

To run the training you have to edit `backend:megatron:data_path:` and run the command:

```
cd /flash/project_462001516/users/vitiugin/oellm-autoexp
export HF_HOME=/flash/project_462001516/users/vitiugin/.cache
export SLURM_ACCOUNT=project_462001516

uv run python scripts/run_autoexp.py --config-name experiments/datamix/scale/*.*B
```