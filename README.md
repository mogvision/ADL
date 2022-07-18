[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/grayscale-image-denoising-on-bsd68-sigma15)](https://paperswithcode.com/sota/grayscale-image-denoising-on-bsd68-sigma15?p=adversarial-distortion-learning-for-medical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/grayscale-image-denoising-on-bsd68-sigma25)](https://paperswithcode.com/sota/grayscale-image-denoising-on-bsd68-sigma25?p=adversarial-distortion-learning-for-medical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/grayscale-image-denoising-on-bsd68-sigma50)](https://paperswithcode.com/sota/grayscale-image-denoising-on-bsd68-sigma50?p=adversarial-distortion-learning-for-medical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/color-image-denoising-on-cbsd68-sigma15)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma15?p=adversarial-distortion-learning-for-medical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/color-image-denoising-on-cbsd68-sigma25)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma25?p=adversarial-distortion-learning-for-medical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/color-image-denoising-on-cbsd68-sigma35)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma35?p=adversarial-distortion-learning-for-medical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adversarial-distortion-learning-for-medical/color-image-denoising-on-cbsd68-sigma50)](https://paperswithcode.com/sota/color-image-denoising-on-cbsd68-sigma50?p=adversarial-distortion-learning-for-medical)

<!--  https://paperswithcode.com/task/color-image-denoising -->

## ADL: Adversarial Distortion Learning for Denoising and Distortion Removal
<!-- [![download](https://img.shields.io/github/downloads/mogvision/ADL/total.svg)](https://github.com/mogvision/ADL/releases) ![visitors](https://visitor-badge.glitch.me/badge?page_id=mogvision/ADL) -->

<!--[Kai Zhang](https://cszn.github.io/) -->
[Morteza Ghahremani](https://scholar.google.com/citations?user=yhXUlXsAAAAJ), [Mohammad Khateri](https://scholar.google.com/citations?user=vHtGWmoAAAAJ&hl=en), [Alejandra Sierra](https://scholar.google.fi/citations?user=cxP2f78AAAAJ&hl=en), [Jussi Tohka](https://scholar.google.com/citations?user=StmRhaUAAAAJ&hl=en)



*[AiVi](https://www.uef.fi/en/unit/ai-virtanen-institute-for-molecular-sciences), UEF, Finland*


[<img src="figs/brainADL.gif" width="180px" height="180px"/>](https://imgsli.com/OTM3OTI)
[<img src="figs/skinADL.gif" width="180px" height="180px"/>](https://imgsli.com/OTM3OTA)
[<img src="figs/emADL.gif" width="180px" height="180px"/>](https://imgsli.com/OTM3ODE)
[<img src="figs/cbsd68ADL.gif" width="250px" height="180px"/>](https://imgsli.com/MTAyNDQ4)

_______
This repository is the official implementation of ADL: Adversarial Distortion Learning for denoising medical and computer vision images ([arxiv](https://arxiv.org/abs/2204.14100), supp, pretrained models, visual results). 

| [TensorFlow <img src="figs/tf.png" width="25"/>](https://github.com/mogvision/ADL/tree/main/TensorFlow) | [PyTorch  <img src="figs/pytorch.png" width="18"/>](https://github.com/mogvision/ADL/tree/main/PyTorch) |  [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> ](https://github.com/mogvision/ADL/blob/main/ADLdemo.ipynb) | 
|:---: |:---: |:---: |
<table align="center"></table>

\
ADL achieves state-of-the-art Gaussian denoising performance in

- grayscale/color image denoising in Medical imaging :fire::fire::fire:
- grayscale/color image denoising in Computer Vision images :fire::fire::fire:
- JPEG compression artifact reduction :fire::fire::fire:
- grayscale/color deblurring :fire::fire::fire:



Network architectures
----------
* Proposed Efficient-UNet (Denoiser)
<img src="figs/Denoiser.PNG" width="780px"/> 

* Proposed Efficient-UNet (Discriminator)
<img src="figs/discriminator.PNG" width="780px"/> 
______________
 
# Denoising Results on [BSD68](https://paperswithcode.com/dataset/bsd) and [CBSD68](https://paperswithcode.com/dataset/cbsd68):

* Results on the [BSD68](https://paperswithcode.com/dataset/bsd) dataset for Additive white Gaussian noise:

|  σ    | BM3D| WNNM | DnCNN  | NLRN | FOCNet | MWCNN | DRUNet | SwinIR | ADL (ours) |
|:-----:|:---:|:----:|:------:|:----:|:------:|:-----:|:-----:|:-----:|:---------:|
| 15 | 31.08 | 31.37 | 31.73 | 31.88  | 31.83 | 31.86 | 31.91 | 31.97 | :fire: **32.11** :fire:|
| 25 | 28.57 | 28.83 | 29.23 | 29.41  | 29.38 | 29.41 | 29.48 | 29.50 | :fire: **29.50** :fire:|
| 50 | 25.60 | 25.87 | 26.23 | 26.47  | 26.50 | 26.53 | 26.59 | 26.58 | :fire: **26.87** :fire:|

- [x] Here we reported the results of the techniques reported by the authors.
- [x] Our ADL was trained on the grey [Flickr2K](https://github.com/LimBee/NTIRE2017) dataset only!

| CBSD68 (img_id: test015)| Noisy (σ=25) | SwinIR  | ADL (ours) |
|    :---      |     :---:    | :-----:|  :-----: | 
| <img width="200" src="figs/gt-test015.png"> | <img width="200" src="figs/noisy-test015.png"> | <img width="200" src="figs/test005_SwinIR.png">|<img width="200" src="figs/test005_ADL.png">




* Results on the [CBSD68](https://paperswithcode.com/dataset/cbsd68) dataset for Additive white Gaussian noise:

| σ | BM3D | WNNM  | EPLL | MLP |  CSF | TNRD  | DnCNN  | DRUNet | SwinIR | ADL (ours) |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| 15 | 33.52 | 33.90 | 33.86 | 33.87 | 33.91 |  -    | 34.10 | 34.30 | 34.42 | :fire: **34.61** :fire:|
| 25 | 30.71 | 31.24 | 31.16 | 31.21 | 31.28 | 31.24 | 31.43 | 31.69 | 31.78 | :fire: **32.18** :fire:|
| 50 | 27.38 | 27.95 | 27.86 | 27.96 | 28.05 | 28.06 | 28.16 | 28.51 | 28.56 | :fire: **29.02** :fire:|

| CBSD68 (img_id: test015)| Noisy (σ=50) | SwinIR  | ADL (ours) |
|    :---      |     :---:    | :-----:|  :-----: | 
| <img width="200" src="figs/gt-test015c.png"> | <img width="200" src="figs/noisy-test015c.png"> | <img width="200" src="figs/test005_SwinIRc.png">|<img width="200" src="figs/test005_ADLc.png">


# Denoising Results on Medical Images:
<p align="center">
<img width="700" src="figs/adl_medical.png">
</p>
<details>
<summary> 2D (click here)</summary>
<p align="center">
  <img width="900" src="figs/2Dmedical.png">
   <img width="900" src="figs/2d.png">
</p>
</details>

<details>
<summary> 3D MRI Brain-BrainWeb (click here)</summary>
<p align="center">
  <img width="500" src="figs/3Dbrain.png">
  <img width="900" src="figs/mribrain.png">
</p>
</details>

<details>
<summary> 3D MRI knee-fastMRI (click here)</summary>
<p align="center">
<img width="500" src="figs/3Dknee.png">
 </p>
</details>



_______
## Citation

If you find ADL useful in your research, please cite our tech report:

```bibtex
@article{ADL2022,
    author = {Morteza Ghahremani, Mohammad Khateri, Alejandra Sierra, Jussi Tohka},
    title = {Adversarial Distortion Learning for Medical Image Denoising},
    journal = {arXiv:2204.14100},
    year = {2022},
}
```
