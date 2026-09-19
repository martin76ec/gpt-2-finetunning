# GPT 2 Small

> [!NOTE]
> I did record the live process of this lab, check it at [youtube](https://youtu.be/_Z2A4Gl0N1Y)

## Pre Requisites

1. uv
2. Data folder with `gpt2-small-124M.pth` and `instruction-data.json` the .pth file can be downloaded from [here](!https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth)

## Reproduction Steps

1. Run `uv sync` to install dependencies
2. Run `make run` to train the model and save into outputs/gpt2-tunned.pth
