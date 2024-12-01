import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def get_calib_dataset_openvla(
    data="pileval", tokenizer=None, n_samples=512, block_size=512
):
    import os

    dataset_dir = ("/mnt/align4_drive/rachelm8/tinyml/openvla-7b+calib_feat+b1--original/multimodal_embeddings"
        #"/sota/openvla/ckpt/openvla-7b+calib_feat+b1--original/multimodal_embeddings"
    )
    n_items = len(os.listdir(dataset_dir))
    samples = []
    n_run = 0
    for batch_idx in range(n_items):
        embed = torch.load(os.path.join(dataset_dir, f"{batch_idx}.pt"))
        sample = embed
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
