# 🗝 GeoEdit

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://PRIS-CV.github.io/GeoEdit/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.08388-b31b1b.svg)](https://arxiv.org/abs/2602.08388)

<!-- <p align = "center">
<img  src="static\images\logo_transparent.png" width="150" />
</p> -->
Code release for "Geometric Image Editing via Effects-Sensitive In-Context Inpainting with Diffusion Transformers"


**Abstract**: Recent advances in diffusion models have significantly improved image editing. However, challenges persist in handling geometric transformations, such as translation, rotation, and scaling, particularly in complex scenes. Existing approaches suffer from two main limitations: (1) difficulty in achieving accurate geometric editing of object translation, rotation, and scaling; (2) inadequate modeling of intricate lighting and shadow effects, leading to unrealistic results. To address these issues, we propose GeoEdit, a framework that leverages in-context generation through a diffusion transformer module, which integrates geometric transformations for precise object edits. Moreover, we introduce Effects-Sensitive Attention, which enhances the modeling of intricate lighting and shadow effects for improved realism. To further support training, we construct RS-Objects, a large-scale geometric editing dataset containing over 120,000 high-quality image pairs, enabling the model to learn precise geometric editing while generating realistic lighting and shadows. Extensive experiments on public benchmarks demonstrate that GeoEdit consistently outperforms state-of-the-art methods in terms of visual quality, geometric accuracy, and realism.
## News

- **Feb 9 2026**: 🔥 Our [Project Page](https://sure66666.github.io/GeoEdit/) has been published!
- **Jan 26 2026**: 🔥 🔥 GeoEdit has been accepted to ICLR'26!

|     | TODO Task     | Update                   |
| --- | ------------- | ------------------------ |
| ☐   | 📂 **Dataset** | Under active preparation |
| ☐   | ⚖️ **Model Weights** | Under active preparation |

<!-- ## Citation

If you find this code repository useful in your research, please consider citing our paper:

```
@article{wei2025omnieraserremoveobjectseffects,
title={OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data},
author={Runpu Wei and Zijin Yin and Shuo Zhang and Lanxiang Zhou and Xueyi Wang and Chao Ban and Tianwei Cao and Hao Sun and Zhongjiang He and Kongming Liang and Zhanyu Ma},
journal={arXiv preprint arXiv:2501.07397},
year={2025},
url={https://arxiv.org/abs/2501.07397},
}
``` -->



## 🛠️ Installation

```shell
conda create -n geoedit python==3.10

conda activate geoedit

pip install -r requirements.txt
```


## 🚀 Training

* **Update FLUX model paths:** In `experiments/experiments.yaml`, replace `/path/to/black-forest-labs-FLUX.1-Fill-dev` and `/path/to/black-forest-labs-FLUX.1-Redux-dev` with your actual paths.  

* **Run the training script:** Execute the training using the command:  

    ```bash
    bash scripts/train.sh
    ```

