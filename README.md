language: en
license: mit
library_name: transformers
tags:
- text-classification
- sentiment-analysis
- financial-news
- distilbert
datasets:
- ankurzing/sentiment-analysis-for-financial-news
pipeline_tag: text-classification
---

# DistilBERT Fine-tuned for Financial News Sentiment Analysis
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%7C%20Datasets-yellow.svg) ![Transformers](https://img.shields.io/badge/Transformers-4.x-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

## 📜 Model Description

This is a `distilbert-base-uncased` model fine-tuned for sentiment analysis on financial news headlines. The model classifies a given headline into one of three categories: **positive**, **neutral**, or **negative**.

This project was developed as a demonstration of fine-tuning a Transformer model for a specific domain task. The entire training process was conducted in a Google Colab environment, utilizing datasets from Kaggle and leveraging the Hugging Face ecosystem for training and deployment.

- **Base Model:** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- **Fine-tuning Task:** Sentiment Analysis (Sequence Classification)
- **Language:** English
- **Developed by:** David Nascimento

## 🚀 How to Use

You can easily use this model with the `pipeline` function from the `transformers` library.

```python
from transformers import pipeline

# Carrega o modelo do Hugging Face Hub
model_name = "David-Cunha/distilbert-base-uncased-finetuned-financial-news-sentiment"
sentiment_analyzer = pipeline("text-classification", model=model_name)

# Exemplos de uso
headline1 = "Stock market reaches all-time high as tech giants soar."
result1 = sentiment_analyzer(headline1)
print(f"Headline: '{headline1}'\nSentiment: {result1[0]['label']} (Score: {result1[0]['score']:.4f})\n")
# Saída esperada: positive

headline2 = "The company announced a delay in its quarterly earnings report."
result2 = sentiment_analyzer(headline2)
print(f"Headline: '{headline2}'\nSentiment: {result2[0]['label']} (Score: {result2[0]['score']:.4f})\n")
# Saída esperada: neutral or negative

headline3 = "Profits fall sharply amid rising inflation concerns."
result3 = sentiment_analyzer(headline3)
print(f"Headline: '{headline3}'\nSentiment: {result3[0]['label']} (Score: {result3[0]['score']:.4f})\n")
# Saída esperada: negative
```

## 💻 Training Procedure
The model was fine-tuned on the "Sentiment Analysis for Financial News" dataset. This dataset was created by Ankur Zing and is publicly available on Kaggle. We are grateful for this contribution to the community.

Dataset Source: Sentiment Analysis for Financial News on Kaggle(https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

Preprocessing:
The original dataset contains two columns: the sentiment (positive, neutral, negative) and the text. The labels were mapped to integer IDs as follows:

neutral: 0

positive: 1

negative: 2

The dataset was then split into a training set (80%) and a test set (20%). The texts were tokenized using the standard distilbert-base-uncased tokenizer.

Hyperparameters
The model was trained using the transformers Trainer API with the following hyperparameters:
```python
TrainingArguments(
    output_dir="distilbert-sentiment-finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2, # Apenas 2 épocas para um treinamento rápido
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
)
```

## 📊 Evaluation Results
The model achieved the following performance on the test set (20% of the data). Remember to replace these placeholder values with your actual results.
Metric	  |   Value
----------------------
Accuracy	|   0.92
______________________
Loss	    |   0.23
______________________

Citation
If you use this model in your work, please consider citing it:
@misc{david_cunha_distilbert_financial_sentiment_2025,
  author = {David Nascimento},
  title = {DistilBERT Fine-tuned for Financial News Sentiment Analysis},
  year = {2025},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{[https://huggingface.co/David-Cunha/distilbert-base-uncased-finetuned-financial-news-sentiment](https://huggingface.co/David-Cunha/distilbert-base-uncased-finetuned-financial-news-sentiment)}}
}
