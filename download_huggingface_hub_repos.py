from huggingface_hub import snapshot_download

repo_name = "monsoon-nlp/BioMedGPT-16bit"
path_to_save = "/projects/frink/models/BioMedGPT-16bit"

snapshot_download(repo_name, local_dir=path_to_save)

print("Download complete!!")
