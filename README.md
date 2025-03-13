# ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5

[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow.svg)](https://huggingface.co/datasets/BAAI/ChildMandarin)
[![Hugging Face Datasets](https://img.shields.io/badge/Paper-link-orange)](https://arxiv.org/abs/2409.18584)
[![License: CC BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--SA--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
## ⭐ Introduction

This repository contains the **ChildMandarin** dataset, a comprehensive Mandarin speech dataset specifically designed for young children aged 3 to 5. This dataset aims to address the scarcity of resources in this area and facilitate research in child speech recognition, speaker verification, and related fields.

## 🚀 Dataset Details

*   **Age Range:** 3-5 years old
*   **Total Duration:** 41.25 hours
*   **Number of Speakers:** 397
*   **Geographic Coverage:** 22 out of 34 provincial-level administrative divisions in China
*   **Gender Distribution:** Balanced across all age groups
*   **Recording Devices:** Smartphones (Android and iPhone)
*   **Recording Environment:** Quiet indoor environments
*   **Annotation:** Character-level manual transcriptions, age, gender, birthplace, device, accent level.
* **Content:** Unrestricted, focusing on age-appropriate daily communication.
* **Data Format:** WAV PCM, 16kHz sampling rate, 16-bit precision

### Dataset Statistics

| Split | # Speakers | # Utterances | Duration (hrs) | Avg. Utterance Length (s) |
| :---- | :--------: | :----------: | :------------: | :-----------------------: |
| Train |    317     |    32,658    |     33.35      |           3.68            |
| Dev   |     39     |    4,057     |      3.78      |           3.35            |
| Test  |     41     |    4,198     |      4.12      |           3.53            |
| **Sum**|  **397**   |  **40,913**    |   **41.25**     |       **3.52**           |

More details could be found in our paper [ChildMandarin](https://arxiv.org/abs/2409.18584)

## 📐 Experiments

We conducted experiments on Automatic Speech Recognition (ASR) and Speaker Verification (SV) tasks to evaluate the dataset.

### 1️⃣ ASR Results

#### Models Trained from Scratch

| Encoder     | Loss       | # Params | Greedy | Beam  | Attention | Attention Rescoring |
| :---------- | :--------- | :------- | :----- | :---- | :-------- | :------------------ |
| Transformer | CTC+AED    | 29M      | 34.55  | 34.4  | 40.61     | 32.15               |
| Conformer   | CTC+AED    | 31M      | 28.73  | 28.72 | 31.60     | 27.38               |
| Conformer   | RNN-T+AED  | 45M      | 37.11  | 37.14 | 33.84     | 37.14               |
| Paraformer  | Paraformer | 30M      | 31.86  | 28.94 | -         | -                   |

#### Fine-tuned Pre-trained Models

| Model           | # Params | Zero-shot | Fine-tuning |
| :-------------- | :------- | :-------- | :---------- |
| CW              | 122M     | 18.05     | 13.66       |
| Whisper-tiny    | 39M      | 67.63     | 28.78       |
| Whisper-base    | 74M      | 51.49     | 23.33       |
| Whisper-small   | 244M     | 37.99     | 17.45       |
| Whisper-medium  | 769M     | 28.55     | 18.97       |
| Whisper-large-v2| 1,550M   | 29.43     | -           |


#### More Pre-trained Models

| Model           | # Params | Zero-shot | 
| :-------------- | :------- | :-------- | 
| Qwen-Audio      |  7.7B   | 20.39     | 
| Qwen2-Audio    |  8.2B   | 11.54     | 
| SenseVoice (Small)    |   234M   | 11.89     | 


### 2️⃣ SV Results
|      Model      | # Params | Dim | Dev (%) | EER (%) | minDCF | EER (%) | minDCF  |
|:---------------:|:--------:|:---:|:-------:|:-------:|:------:|:-------:|:-------:|
|    x-vector    |   4.2M   | 512 |   75.4  |  8.91  | 0.7198 |  25.92 |  0.9780  |
|  ECAPA-TDNN   |  20.8M   | 192 |   84.6  | 13.72 | 0.8697 | 27.77  | 0.9490 |
| ResNet-TDNN |   15.5M  |  256  |  91.9  |   9.57  | 0.6597 | 22.11  | 0.9044 |



## 🤗 HuggingFace Dataset

You can access the ChildMandarin dataset on Hugging Face Datasets:

[https://huggingface.co/datasets/BAAI/ChildMandarin](https://huggingface.co/datasets/BAAI/ChildMandarin)



##  📚 Cite me
```
@article{zhou2024childmandarin,
  title={ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5},
  author={Zhou, Jiaming and Wang, Shiyao and Zhao, Shiwan and He, Jiabei and Sun, Haoqin and Wang, Hui and Liu, Cheng and Kong, Aobo and Guo, Yujie and Qin, Yong},
  journal={arXiv preprint arXiv:2409.18584},
  year={2024}
}
``` 