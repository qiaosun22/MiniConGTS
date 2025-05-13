[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/minicongts-a-near-ultimate-minimalist/aspect-sentiment-triplet-extraction-on-aste)](https://paperswithcode.com/sota/aspect-sentiment-triplet-extraction-on-aste?p=minicongts-a-near-ultimate-minimalist)

# MiniConGTS: A Near Ultimate Minimalist Contrastive Grid Tagging Scheme for Aspect Sentiment Triplet Extraction

## Update
2025-03-30:

Fixed the missing filesc for your fast installing the reliances and getting started.

Now you can enjoy this repo by simply clone and run 

```bash
conda env create -f environment.yml
python main.py
```

Have fun!

## Overview

This repository serves as the official codebase for my recent work, "MiniConGTS: A Near Ultimate Minimalist Contrastive Grid Tagging Scheme for Aspect Sentiment Triplet Extraction". The project implements a minimalist tagging scheme and a novel token-level contrastive learning strategy to enhance aspect sentiment triplet extraction performance. The approach leverages the power of Pretrained Language Models (PLMs), such as BERT, RoBERTa etc. to achieve state-of-the-art results without relying on complex classification head designs or external semantic enhancements.

[Arxiv Preprint](https://arxiv.org/abs/2406.11234)

![image](https://github.com/qiaosun22/MiniConGTS/assets/136222260/ad019d55-7d90-4299-a53a-c980b80e4e49)

![image](https://github.com/qiaosun22/MiniConGTS/assets/136222260/762a1cfb-3de3-46c1-8249-7c2c5fa51e84)

![image](https://github.com/qiaosun22/MiniConGTS/assets/136222260/b1dd1499-282b-4089-aa21-b08bf567ac5f)

![image](https://github.com/qiaosun22/MiniConGTS/assets/136222260/94e92fa4-c61e-4b5a-8986-fd2177148f25)

## Key Features

- **Minimalist Grid Tagging Scheme**: Uses the fewest classes of labels to simplify the tagging process.
- **Token-level Contrastive Learning**: Improves the representation quality of PLMs, enhancing their internal potential.
- **High Performance**: Achieves state-of-the-art results in Aspect-Based Sentiment Analysis (ABSA) with minimal reliance on complex architectures.
- **Evaluation on GPT Models**: Includes first-time evaluations of the Chain-of-Thought method in context learning scenarios.
- **Comprehensive Analysis**: Provides proofs and theoretical analyses to support the efficacy of the proposed methods.

## Repository Structure

```
MiniConGTS/
│
├── data/                   # Directory for datasets
│   ├── D1/
│   │   ├── res14/
│   │   │   ├── train.json
│   │   │   ├── dev.json
│   │   │   └── test.json
│   │   └── ...             # Other datasets
│   └── D2/
│       └── ...
│
├── modules/
│   ├── models/
│   │   ├── roberta.py      # RoBERTa model definition
│   │   └── saved_models/   # Directory for saving trained models
│   └── f_loss.py           # Focal loss implementation
│
├── tools/
│   └── trainer.py          # Training and evaluation script
│
├── utils/
│   ├── common_utils.py     # Utility functions
│   ├── data_utils.py       # Data loading and preprocessing
│   └── ...
│
├── main.py                 # Main script for training and evaluation
├── requirements.txt        # List of required packages
└── README.md               # This file
```

## Getting Started

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/MiniConGTS.git
   cd MiniConGTS
   ```
2. **Install dependencies:**

   ```bash
   conda env create -f environment.yml
   ```

### Preparing the Data

1. **Place your datasets** in the `./data/` directory following the provided structure.

### Training the Model

1. **Configure training parameters** in the `main.py` script or pass them as command-line arguments.
2. **Run the training script:**

   ```bash
   python main.py --max_sequence_len 100 --batch_size 16 --epochs 2000 --dataset res14
   ```

   for more parameter setting:

   ```
   python main.py \
    --max_sequence_len 100 \
    --sentiment2id "{'negative': 2, 'neutral': 3, 'positive': 4}" \
    --model_cache_dir "./modules/models/" \
    --model_name_or_path "roberta-base" \
    --batch_size 16 \
    --device "cuda" \
    --prefix "./data/" \
    --data_version "D1" \
    --dataset "res14" \
    --bert_feature_dim 768 \
    --epochs 2000 \
    --class_num 5 \
    --task "triplet" \
    --model_save_dir "./modules/models/saved_models/" \
    --log_path "./logs/training_log.log" \
    --learning_rate 1e-3 \
    --warmup_steps 500 \
    --weight_decay 0.01

   ```

Alternatively, you can start your jupyter kernal and debug each intermidiate step easily in a notebook using:

```
main.ipynb
```

### Evaluation

The model is evaluated on the test set after training. The results, including metrics such as accuracy, precision, recall, and F1-score, are logged to a file specified in the `log_path`.

## Citation

If you use this code in your research, please cite the paper as follows:

```
@inproceedings{sun-etal-2024-minicongts,
    title = "{M}ini{C}on{GTS}: A Near Ultimate Minimalist Contrastive Grid Tagging Scheme for Aspect Sentiment Triplet Extraction",
    author = "Sun, Qiao  and
      Yang, Liujia  and
      Ma, Minghao  and
      Ye, Nanyang  and
      Gu, Qinying",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.165/",
    doi = "10.18653/v1/2024.emnlp-main.165",
    pages = "2817--2834",
    abstract = "Aspect Sentiment Triplet Extraction (ASTE) aims to co-extract the sentiment triplets in a given corpus. Existing approaches within the pretraining-finetuning paradigm tend to either meticulously craft complex tagging schemes and classification heads, or incorporate external semantic augmentation to enhance performance. In this study, we, for the first time, re-evaluate the redundancy in tagging schemes and the internal enhancement in pretrained representations. We propose a method to improve and utilize pretrained representations by integrating a minimalist tagging scheme and a novel token-level contrastive learning strategy. The proposed approach demonstrates comparable or superior performance compared to state-of-the-art techniques while featuring a more compact design and reduced computational overhead. Additionally, we are the first to formally evaluate GPT-4`s performance in few-shot learning and Chain-of-Thought scenarios for this task. The results demonstrate that the pretraining-finetuning paradigm remains highly effective even in the era of large language models."
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the authors and contributors of the `transformers` library by Hugging Face.
