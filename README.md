# grla-rl

conda create -n grla-rl python=3.10
conda activate grla-rl

pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install notebook ipykernel