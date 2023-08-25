# SNARE (a multimodal alignment probing benchmark)
Project for our work "[<b>Can Linguistic Knowledge Improve Multimodal Alignment in Vision-Language Pre-training?</b>](https://arxiv.org/abs/2308.12898)", which is the <b>first large-scale multimodal alignment probing benchmark</b>, to detect the vital linguistic components in the vision-language pretrained models.

SNARE contains four tasks: 1) semantic structure, 2) negation logic, 3) attribute ownership, and 4) relationship composition.  

![avatar](./img/snare.png)

## Abstract
The multimedia community has shown a significant interest in perceiving and representing the physical world with multimodal pretrained neural network models, and among them, the visual-language pertaining (VLP) is, currently, the most captivating topic. The common practice for pretraining the visual-language backbone is supervising the training process with paired image-text data. However, there have been few endeavors dedicated to the exploration of 1) whether essential linguistic knowledge (e.g., semantics and syntax) can be extracted during VLP, and 2) how such linguistic knowledge impact or enhance the multimodal alignment. In response, here we aim to elucidate the impact of comprehensive linguistic knowledge, including semantic expression and syntactic structure, on multimodal alignment. Specifically, we design and release the SNARE, the first large-scale multimodal alignment probing benchmark, to detect the vital linguistic components, e.g., lexical, semantic, and syntax knowledge, containing four tasks: Semantic structure, Negation logic, Attribute ownership, and Relationship composition. Based on our proposed probing benchmark, our holistic analyses of five advanced VLP models (i.e., BLIP, CLIP, Flava, X-VLM, and BLIP2) illustrate that the VLP model: <i>i)</i> shows insensitivity towards complex syntax structures and relies on content words for sentence comprehension; <i>ii)</i> demonstrates limited comprehension of combinations between sentences and negations; <i>iii)</i> faces challenges in determining the presence of actions or spatial relationships within visual information and struggles with verifying the correctness of triple combinations. Given the above findings, we suggest that, to improve the multimodal alignment, 1) using the large generative language model as the language backbone in VLP to understand complex sentences; 2) establishing high-quality datasets by highlighting the content words and using simple syntax (e.g., short-distance semantic composition) to improve multimodal alignment; and 3) incorporating more fine-grained visual knowledge (e.g., spatial relationships) into pretraining objectives.

## Citation
If you find our work helpful, please consider citing as follows:  
```ruby
@article{Wang2023SNARE,
  title={Can Linguistic Knowledge Improve Multimodal Alignment in Vision-Language Pretraining?},
  author={Fei Wang and Liang Ding and Jun Rao and Ye Liu and Li Shen and Changxing Ding},
  year={2023},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2308.12898}
}
```
