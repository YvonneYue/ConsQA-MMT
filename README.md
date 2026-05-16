# ConsQA-MMT

Multimodal Machine Translation with Question Enhancement and Answer Consistency Constraint (基于问题增强和回答能力一致性约束的多模态机器翻译)

## 项目结构

```
ConsQA-MMT/
├── fairseq/                    # 修改版 FairSeq 框架
│   ├── criterions/             # 损失函数（含 consqa_mmt 一致性损失）
│   ├── data/                   # 数据集定义（含多模态数据加载）
│   ├── models/transformer/     # 模型定义（含 DualSAEncoder + DualLayersDecoder）
│   ├── modules/                # 模块（含 SelectiveAttention）
│   └── tasks/                  # 任务定义（consqa_mmt）
├── fairseq_cli/                # FairSeq CLI 入口（train/generate/preprocess）
├── scripts/                    # 工具脚本（checkpoint 平均等）
├── data-bin/                   # 预处理后的二进制数据
│   ├── multi30k.en-de.consqa_mmt/
│   ├── multi30k.en-fr.consqa_mmt/
│   └── pretrain.en-de.consqa_mmt/
├── raw_image/                  # 图片数据
│   ├── flickr30k-images/       # Flickr30K 训练+test2016 图片
│   ├── test2017/               # Multi30K test2017 图片
│   └── testcoco/               # COCO 测试图片
├── vit-mae-base/               # ViT-MAE 预训练视觉模型
├── data/vit-mae-base/          # 预计算的图像特征缓存（自动生成）
├── checkpoints/                # 训练产出的模型权重
├── mosesdecoder/               # Moses tokenizer
├── subword-nmt-master/         # BPE 分词工具
├── train_enfr.sh               # En→Fr 训练脚本
├── test_enfr.sh                # En→Fr 测试脚本
├── translate.sh                # 通用翻译脚本
├── rerank.py                   # 翻译结果排序
├── meteor.py                   # METEOR 指标计算
├── setup.py                    # 安装配置
└── README.md
```

## 环境配置

### 依赖

- Python 3.10
- PyTorch >= 2.0（含 CUDA 支持）
- transformers >= 5.0
- sacrebleu
- omegaconf
- hydra-core >= 1.3

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n mmt python=3.10 -y
conda activate mmt

# 2. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install transformers sacrebleu omegaconf hydra-core numpy pillow tqdm

# 4. 安装 fairseq（editable 模式，跳过 C 扩展）
cd ConsQA-MMT
READTHEDOCS=1 pip install -e . --no-deps --no-build-isolation
```

### 数据准备

1. 将 Flickr30K 图片放入 `raw_image/flickr30k-images/`
2. 将 test2017 图片放入 `raw_image/test2017/`
3. 将 testcoco 图片放入 `raw_image/testcoco/`
4. 确保 `data-bin/` 下各目录中的 `.imgname` 文件路径指向正确的图片位置

### 下载 ViT-MAE 模型

```bash
# 如果 vit-mae-base/ 目录为空，从 HuggingFace 下载
huggingface-cli download facebook/vit-mae-base --local-dir vit-mae-base/
```

## 训练

### En→Fr 训练

```bash
bash train_enfr.sh
```

### En→De 训练

修改 `train_enfr.sh` 中的参数：
```bash
src_lang=en
tgt_lang=de
data_dir=multi30k.en-de.consqa_mmt
tag=en-de-consqa_mmt_0.001_2048_0.2w1
```

### 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| arch | transformer_consqa_mmt | 模型架构 |
| criterion | consqa_mmt_label_smoothed_cross_entropy | 损失函数 |
| lr | 0.001 | 学习率 |
| max-tokens | 2048 | 批量大小（token 数） |
| warmup | 4000 | 学习率预热步数 |
| weight1 | 0.2 | VQA 损失权重 |
| weight2 | 0.1 | 一致性损失权重 |
| patience | 10 | 早停 |
| keep-last-epochs | 10 | 保留最近 N 个 checkpoint |
| vision-model | vit-mae-base | 视觉模型路径 |

### 监控训练

```bash
tail -f checkpoints/<tag>/train.log
```

## 测试/推理

### En→Fr 测试

```bash
bash test_enfr.sh
```

脚本会：
1. 对最近 10 个 checkpoint 做平均（ensemble）
2. 用 ensemble 模型在测试集上生成翻译
3. 输出 BLEU 分数

## 算法说明

本项目在标准 Transformer 翻译模型基础上引入：

1. **视觉特征融合**：通过 SelectiveAttention 模块将 ViT-MAE 提取的图像特征与文本编码融合（门控机制）
2. **VQA 辅助任务**：对源语言和目标语言分别构建视觉问答分支，增强模型对图像内容的理解
3. **一致性约束**：约束源/目标语言 VQA 预测之间的距离与标签距离一致
4. **双路径训练**：同时训练 MMT（有图像）和 NMT（无图像）路径，通过 KL 散度约束两者输出分布

总损失：`L = L_mt + w1 * L_vqa + λ(epoch) * L_consistency`

其中 `L_mt = (L_mmt + L_nmt) / 2 + 0.5 * KL(mmt || nmt)`
