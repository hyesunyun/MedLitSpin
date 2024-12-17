from huggingface_hub import snapshot_download

repo_name = "stabilityai/stable-diffusion-2-1"
path_to_save = "./outputs/stable_diffusion_2_1"

snapshot_download(repo_name, local_dir=path_to_save)

# python -m pip install huggingface_hub