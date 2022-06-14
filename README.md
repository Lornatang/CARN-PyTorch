# CARN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network](https://arxiv.org/abs/1803.08664v5)
.

## Table of contents

- [CARN-PyTorch](#carn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network](#fast-accurate-and-lightweight-super-resolution-with-cascading-residual-network)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 31: `upscale_factor` change to `2`.
- line 33: `mode` change to `test`.
- line 70: `model_path` change to `results/pretrained_models/CARN_x2-DIV2K-2096ee7f.pth.tar`.

### Train model

- line 31: `upscale_factor` change to `2`.
- line 33: `mode` change to `train`.
- line 35: `exp_name` change to `CARN_x2`.

### Resume train model

- line 31: `upscale_factor` change to `2`.
- line 33: `mode` change to `train`.
- line 35: `exp_name` change to `CARN_x2`.
- line 48: `resume` change to `samples/CARN_x2/epoch_xxx.pth.tar`.

## Result

Source of original paper results: [https://arxiv.org/pdf/1803.08664v5.pdf](https://arxiv.org/pdf/1803.08664v5.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

| Method | Scale |          Set5 (PSNR/SSIM)           |          Set14 (PSNR/SSIM)          |         BSD100 (PSNR/SSIM)          |        Urban100 (PSNR/SSIM)         |
|:------:|:-----:|:-----------------------------------:|:-----------------------------------:|:-----------------------------------:|:-----------------------------------:|
|  CARN  |   2   | 37.76(**37.80**)/0.9590(**0.9605**) | 33.52(**33.34**)/0.9166(**0.9159**) | 32.09(**32.04**)/0.8978(**0.8988**) | 31.92(**31.48**)/0.9256(**0.9220**) |
|  CARN  |   3   | 34.29(**34.16**)/0.9255(**0.9260**) | 30.29(**30.08**)/0.8407(**0.8381**) | 29.06(**28.97**)/0.8034(**0.8034**) | 28.06(**27.72**)/0.8493(**0.8432**) |
|  CARN  |   4   | 32.13(**32.02**)/0.8937(**0.8940**) | 28.60(**28.45**)/0.7806(**0.7792**) | 27.58(**27.50**)/0.7349(**0.7351**) | 26.07(**25.81**)/0.7837(**0.7775**) |

```bash
# Download `CARN_x2-DIV2K-4797e51b.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python ./inference.py --inputs_path ./figure/comic_lr.png --output_path ./figure/comic_sr.png --weights_path ./results/pretrained_models/CARN_x2-DIV2K-4797e51b.pth.tar
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic_lr.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/comic_sr.png"/></span>

```text
Build CARN model successfully.
Load CARN model weights `./results/pretrained_models/CARN_x2-DIV2K-4797e51b.pth.tar` successfully.
SR image save to `./figure/comic_sr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network

_Namhyuk Ahn, Byungkon Kang, Kyung-Ah Sohn_ <br>

**Abstract** <br>
. In recent years, deep learning methods have been successfully applied to single-image super-resolution tasks. Despite
their great performances, deep learning methods cannot be easily applied to realworld applications due to the
requirement of heavy computation. In this paper, we address this issue by proposing an accurate and lightweight deep
network for image super-resolution. In detail, we design an architecture that implements a cascading mechanism upon a
residual network. We also present variant models of the proposed cascading residual network to further improve
efficiency. Our extensive experiments show that even with much fewer parameters and operations, our models achieve
performance comparable to that of state-of-the-art methods.

[[Paper]](https://arxiv.org/pdf/1803.08664v5.pdf)

```bibtex
@article{DBLP:journals/corr/abs-1803-08664,
  author    = {Namhyuk Ahn and
               Byungkon Kang and
               Kyung{-}Ah Sohn},
  title     = {Fast, Accurate, and, Lightweight Super-Resolution with Cascading Residual
               Network},
  journal   = {CoRR},
  volume    = {abs/1803.08664},
  year      = {2018}
}
```
