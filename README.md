# 📰 Fake News Detection using Bidirectional LSTM

> 🧠 An NLP-based deep learning project that uses a **Bidirectional LSTM (BiLSTM)** network to classify news articles as fake or real.

## 📌 Project Overview

This project implements a **Fake News Classifier** using Natural Language Processing (NLP) and a Bidirectional Long Short-Term Memory (**BiLSTM**) neural network.

The project builds an end-to-end text classification pipeline:

**Raw News Data → Cleaning → Text Preprocessing → Stemming → One-Hot Representation → Padding → Embedding → BiLSTM → Binary Classification**

---

## 🎯 Objective

The main objective is to classify news content into two categories:

- 🔴 **Fake News** — label `1`
- 🟢 **Real News** — label `0`

The model learns patterns from the **news titles** and uses them for binary classification.

---

## 📊 Dataset

The project uses the Kaggle **Fake News** dataset.

The dataset contains columns such as:

- `id`
- `title`
- `author`
- `text`
- `label`

The notebook initially loads **20,800 records** and, after removing rows containing missing values, works with **18,285 records**.

---

## 🧹 Data Preprocessing

The project performs the following preprocessing steps:

1. 🗂️ Load the dataset using Pandas.
2. 🔍 Check for missing values.
3. 🧹 Remove rows containing missing values.
4. 🎯 Separate independent features and target labels.
5. 📰 Extract the news title.
6. 🔡 Convert text to lowercase.
7. 🧹 Remove non-alphabetic characters.
8. 🛑 Remove English stopwords using NLTK.
9. 🌱 Apply Porter Stemming.
10. 🔢 Convert words into integer representations using one-hot encoding.
11. 📏 Pad sequences to a fixed length of **20 tokens**.

---

## 🧠 Model Architecture

The core model is a **Bidirectional LSTM (BiLSTM)** built using TensorFlow/Keras.

```text
Input Sequence
      │
      ▼
Embedding Layer
Vocabulary = 5000
      │
      ▼
Bidirectional LSTM
64 Units
      │
      ▼
Dense Layer
Sigmoid Activation
      │
      ▼
Binary Prediction
