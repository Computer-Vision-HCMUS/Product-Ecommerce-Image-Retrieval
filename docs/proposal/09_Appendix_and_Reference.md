# 09. Appendix and References

## 9.1. References từ local papers

1. N. Venkatesan, M. Suresh, Vethamuthu Richard Paul, Diganta Kumar Das, P. Vijayakumar. **The Rise of Visual Search in E-Commerce: Leveraging AI to Redefine Product Discovery**. Journal of Marketing & Social Research, 2025.
2. Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Michael C. Kampffmeyer, Xiao-Yong Wei, Minlong Lu, Yaowei Wang, Xiaodan Liang. **M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining**. 2022.
3. Prajit Nadkarni, Narendra Varma Dasararaju. **Visually Similar Products Retrieval for Shopsy**. arXiv:2210.04560, 2022.
4. Chang Liu, Peng Hou, Anxiang Zeng, Han Yu. **Transformer-Empowered Multi-Modal Item Embedding for Enhanced Image Search in E-commerce**. AAAI, 2024.
5. Hao Jiang, Haoxiang Zhang, Qingshan Hou, Chaofeng Chen, Weisi Lin, Jingchang Zhang, Annan Wang. **MRSE: An Efficient Multi-modality Retrieval System for Large Scale E-commerce**.
6. Peng Yuan, Bingyin Mei, Hui Zhang. **FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data**. arXiv:2604.10297, 2026.

## 9.2. References được trích trong methodology

7. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. IEEE TPAMI, 2017.
8. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **Deep Residual Learning for Image Recognition**. CVPR, 2016.
9. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL, 2019.
10. Alec Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML, 2021.
11. Florian Schroff, Dmitry Kalenichenko, James Philbin. **FaceNet: A Unified Embedding for Face Recognition and Clustering**. CVPR, 2015.
12. Diederik P. Kingma, Max Welling. **Auto-Encoding Variational Bayes**. ICLR, 2014.
13. Yury A. Malkov, Dmitry A. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs**. IEEE TPAMI, 2020.
14. Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar. **Accelerating Large-Scale Inference with Anisotropic Vector Quantization**. ICML, 2020.
15. Jeff Johnson, Matthijs Douze, Herve Jegou. **Billion-scale Similarity Search with GPUs**. IEEE Transactions on Big Data, 2019.
16. Herve Jegou, Matthijs Douze, Cordelia Schmid. **Product Quantization for Nearest Neighbor Search**. IEEE TPAMI, 2011.

## 9.3. Tool/library references

17. Meta AI. **Faiss Documentation**. https://faiss.ai/
18. Meta AI. **Faiss Guidelines to Choose an Index**. https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
19. Google Research. **ScaNN: Scalable Nearest Neighbors**. https://github.com/google-research/google-research/tree/master/scann
20. Qdrant. **Indexing Documentation**. https://qdrant.tech/documentation/manage-data/indexing/
21. Milvus. **In-memory Index Documentation**. https://milvus.io/docs/index.md
22. NVIDIA. **Apex: Tools for Easy Mixed Precision and Distributed Training in PyTorch**. https://github.com/NVIDIA/apex
23. PyTorch Contributors. **PyTorch**. https://pytorch.org/
24. Hugging Face. **BERT model documentation in Transformers**. https://huggingface.co/docs/transformers/model_doc/bert
25. Hugging Face. **Transformers library**. https://github.com/huggingface/transformers
26. TorchVision. **Detection models and operators**. https://pytorch.org/vision/stable/models.html
27. Librosa. **librosa.feature.mfcc documentation**. https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
28. PyAV Contributors. **PyAV Documentation**. https://pyav.org/docs/stable/
29. FFmpeg Developers. **FFmpeg Documentation**. https://ffmpeg.org/documentation.html
30. Pandas Contributors. **pandas Documentation**. https://pandas.pydata.org/docs/
31. Ross Wightman. **timm: PyTorch Image Models**. https://github.com/huggingface/pytorch-image-models

## 9.4. Paper references for tools and model components

32. Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang. **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering**. CVPR, 2018.
33. Ranjay Krishna et al. **Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations**. International Journal of Computer Vision, 2017.
34. Ashish Vaswani et al. **Attention Is All You Need**. NeurIPS, 2017.
35. Steven Davis, Paul Mermelstein. **Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences**. IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.

## 9.5. Appendix: ký hiệu

| Ký hiệu | Ý nghĩa |
| --- | --- |
| `D(n)` | Dataset gồm n ảnh sản phẩm trong catalog. |
| `q` | Query của người dùng. |
| `K` | Số kết quả trả về. |
| `f(.)` | Feature extractor/encoder. |
| `sim(.)` | Hàm similarity, thường là cosine hoặc inner product sau L2-normalization. |
| ANN | Approximate Nearest Neighbor. |
| JCT | Joint Co-Transformer trong SCALE. |
| SIMCL | Self-harmonized Inter-Modality Contrastive Learning. |
