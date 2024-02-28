# CMMD (Maximum Mean Discrepancy distance using CLIP embeddings)
Unofficial PyTorch implementation of CMMD. 
CMMD is proposed in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). CMMD is considered a more effective metric than FID, designed to overcome the persistent challenges of FID. 

The [original original implementation](https://github.com/google-research/google-research/tree/master/cmmd) is based on JAX and TensorFlow. This implementation replaced them PyTorch and does not need the [`scenic`](https://github.com/google-research/scenic) for computing CLIP embeddings.

* For CLIP embedding, this version uses [`openai/clip-vit-large-patch14-336`](https://huggingface.co/openai/clip-vit-large-patch14-336) from [`huggingface`]

## Installed modules
* torch 1.13.1 (with CUDA 11.7)
* numpy
* [`transformers`](https://github.com/huggingface/transformers)
* pillow
* tqdm


## Run
```bash
python -u cmmd.py --original_folder_path /path/to/original/dataset  --generated_folder_path /path/to/generated/dataset
```

**Notes**:
* if you want to use other version of CLIP: 
```bash
python -u cmmd.py --original_folder_path /path/to/original/dataset  --generated_folder_path /path/to/generated/dataset --model_version model version
```
* if you want to use other subset size:
```bash
python -u cmmd.py --original_folder_path /path/to/original/dataset  --generated_folder_path /path/to/generated/dataset --subset_size subset size
```
## Acknowledgements
Thank you to the authors for proposing CMMD, which improves to evaluate generated data in a robust and reliable manner. 