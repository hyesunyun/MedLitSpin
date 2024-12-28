from huggingface_hub import snapshot_download

repo_name = "meta-llama/Llama-2-13b-chat-hf"
path_to_save = "/projects/frink/models/Llama-2-13b-chat-hf"

snapshot_download(repo_name, local_dir=path_to_save)

print("Download complete!!")
