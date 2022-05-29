# Font Generation with Missing Impression Labels(ICPR2022)
This repository provides PyTorch implementation for [**Font Generation with Missing Impression Labels**](https://arxiv.org/abs/2203.10348). Given an *impression labels*, our proposed model can generate the appropriate style font image. This paper proposes a font generation model that is robust against missing impression labels. 

<img src="figs/proposed_model.png" width=100% alt="The overall structure of the proposed model. The two modules, impression label space compressor (ILSC) and co-occurrence-based missing label
estimator (CMLE), are highlighted because they are newly introduced for missing labels.">

**Note:**
In our other studies, we have also proposed font generation model from specific impression. Please check them from the links below.
- [**Font Generation with Missing Impression Labels**](https://github.com/SeiyaMatsuda/Font-Generation-with-Missing-Impression-Labels) (ICPR 2022): GAN for *impression words*
## Paper
**Font Generation with Missing Impression Labels**.<br>
Seiya Matsuda, Akisato Kimura, and Seichi Uchida<br>
Accepted ICPR2022

[**[Paper]**](https://arxiv.org/abs/2203.10348)

## Abstract
Our goal is to generate fonts with specific impressions, by training a generative adversarial network with a font dataset with impression labels. The main difficulty is that font impression is ambiguous and the absence of an impression label does not always mean that the font does not have the impression. This paper proposes a font generation model that is robust against missing impression labels. The key ideas of the proposed method are (1)a co-occurrence-based missing label estimator and (2)an impression label space compressor. The first is to interpolate missing impression labels based on the co-occurrence of labels in the dataset and use them for training the model as completed label conditions. The second is an encoder-decoder module to compress the high-dimensional impression space into low-dimensional. We proved that the proposed model generates high-quality font images using multi-label data with missing labels through qualitative and quantitative evaluations.
