All of the code are built from the official code of LLaVA-1.5. Mainly, you only need to change several parts.
## Part1
- LLaVA/llava/model/multimodal_projector/builder.py. line290-line309 to decide whether to use the AGV-PR or C-Abstractor. Also line 241 to 243 means (1). use the AGV-PR (2). use the Origin PR (3). skip PR

## Part2
- LLaVA/llava/model/multimodal_encoder/clip_encoder.py. line40-line58 means use our proposed Anchor selector. line60-line73 means use the pooling strategy.

## Part3
- LLaVA/llava/model/llava_arch.py
Pass the selected anchors to the cross attention module. line 142

# Reference
* [LLaVA](https://github.com/haotian-liu/LLaVA)
# Citation

```latex
@misc{liu2024visualanchorsstronginformation,
      title={Visual Anchors Are Strong Information Aggregators For Multimodal Large Language Model}, 
      author={Haogeng Liu and Quanzeng You and Xiaotian Han and Yongfei Liu and Huaibo Huang and Ran He and Hongxia Yang},
      year={2024},
      eprint={2405.17815},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.17815}, 
}
```
