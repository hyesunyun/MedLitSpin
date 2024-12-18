from huggingface_hub import snapshot_download

repo_name = "xz97/AlpaCare-llama2-13b"
path_to_save = "/projects/frink/models/alpacare-13b"

snapshot_download(repo_name, local_dir=path_to_save)

print("Download complete!!")
