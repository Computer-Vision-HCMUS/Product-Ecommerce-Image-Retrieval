# 10. Appendix and References

## 10.1. References từ local papers

1. N. Venkatesan, M. Suresh, Vethamuthu Richard Paul, Diganta Kumar Das, P. Vijayakumar. **The Rise of Visual Search in E-Commerce: Leveraging AI to Redefine Product Discovery**. Journal of Marketing & Social Research, 2025.
2. Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Michael C. Kampffmeyer, Xiao-Yong Wei, Minlong Lu, Yaowei Wang, Xiaodan Liang. **M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining**. 2022.
3. Prajit Nadkarni, Narendra Varma Dasararaju. **Visually Similar Products Retrieval for Shopsy**. arXiv:2210.04560, 2022.
4. Chang Liu, Peng Hou, Anxiang Zeng, Han Yu. **Transformer-Empowered Multi-Modal Item Embedding for Enhanced Image Search in E-commerce**. AAAI, 2024.
5. Hao Jiang, Haoxiang Zhang, Qingshan Hou, Chaofeng Chen, Weisi Lin, Jingchang Zhang, Annan Wang. **MRSE: An Efficient Multi-modality Retrieval System for Large Scale E-commerce**.
6. Peng Yuan, Bingyin Mei, Hui Zhang. **FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data**. arXiv:2604.10297, 2026.

## 10.2. References được trích trong methodology

7. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. IEEE TPAMI, 2017.
8. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **Deep Residual Learning for Image Recognition**. CVPR, 2016.
9. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL, 2019.
10. Alec Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML, 2021.
11. Yury A. Malkov, Dmitry A. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs**. IEEE TPAMI, 2020.
12. Jeff Johnson, Matthijs Douze, Herve Jegou. **Billion-scale Similarity Search with GPUs**. IEEE Transactions on Big Data, 2019.
13. Alexey Dosovitskiy et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**. ICLR, 2021.

## 10.3. Tool/library references

14. Meta AI. **Faiss Documentation**. https://faiss.ai/
15. NVIDIA. **Apex: Tools for Easy Mixed Precision and Distributed Training in PyTorch**. https://github.com/NVIDIA/apex
16. PyTorch Contributors. **PyTorch**. https://pytorch.org/
17. Hugging Face. **BERT model documentation in Transformers**. https://huggingface.co/docs/transformers/model_doc/bert
18. Hugging Face. **Transformers library**. https://github.com/huggingface/transformers
19. TorchVision. **Detection models and operators**. https://pytorch.org/vision/stable/models.html
20. Librosa. **librosa.feature.mfcc documentation**. https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
21. PyAV Contributors. **PyAV Documentation**. https://pyav.org/docs/stable/
22. FFmpeg Developers. **FFmpeg Documentation**. https://ffmpeg.org/documentation.html
23. Pandas Contributors. **pandas Documentation**. https://pandas.pydata.org/docs/

## 10.4. Paper references for tools and model components

24. Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang. **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering**. CVPR, 2018.
25. Ranjay Krishna et al. **Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations**. International Journal of Computer Vision, 2017.
26. Ashish Vaswani et al. **Attention Is All You Need**. NeurIPS, 2017.
27. Steven Davis, Paul Mermelstein. **Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences**. IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.

## 10.5. Appendix: ký hiệu

| Ký hiệu | Ý nghĩa |
| --- | --- |
| `q = (Image, Text, Table, Video, Audio)` | Query đa phương thức; tối thiểu Image hoặc Video. |
| `G = {p_1, ..., p_N}` | Kho sản phẩm. |
| `p_i` | Một listing, cùng bộ năm modality, có thể thiếu nhánh. |
| `K` | Số kết quả trả về. |
| `N` | Số ứng viên lọc thô HNSW, `N > K`. |
| `f(.)` | SCALE encoder, unified embedding. |
| `sim(.)` / `S_emb` | Độ tương đồng embedding (inner product sau L2-normalize). |
| `S` | Alignment score matrix trong SIMCL. |
| `S(u, v)` | Mức bổ trợ semantic giữa hai modality. |
| `S_thuoc_tinh` | Điểm khớp siêu danh mục / danh mục / thông số. |
| `S_tong` | `λ S_emb + (1-λ) S_thuoc_tinh`. |
| `SD`, `D` | Siêu danh mục, danh mục (Hướng 2: `D*` là thương hiệu đa số). |
| `I(·)` | Hàm chỉ thị; 0 nếu thiếu thành phần query. |
| JCT | Joint Cross-modal Transformer. |
| SIMCL | Self-harmonized Inter-Modality Contrastive Learning. |
| HNSW | Hierarchical Navigable Small World (Faiss). |
