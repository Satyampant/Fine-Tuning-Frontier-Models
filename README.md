# Product Pricer: ML-Based Price Estimation from Product Descriptions

A machine learning/Fine Tuning project that estimates product prices from their descriptions using various approaches, from traditional ML models to fine-tuned large language models.

## Overview

This project builds a model that can estimate how much a product costs based solely on its description, features, and metadata. The system is trained on Amazon product data across multiple categories including Electronics, Automotive, Home & Kitchen, and more.

## Dataset

The project uses the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) from HuggingFace, specifically the product metadata.

**Categories included:**
- Automotive
- Electronics
- Office Products
- Tools and Home Improvement
- Cell Phones and Accessories
- Toys and Games
- Appliances
- Musical Instruments

**Dataset Statistics:**
- **Training set:** 400,000 items
- **Test set:** 2,000 items
- **Price range:** $0.50 - $999.49
- **Average price:** ~$60-70 (after balancing)
- **Token range:** 150-180 tokens per item

## Data Curation Process

### 1. Initial Filtering
- Remove items without valid prices
- Filter out items with insufficient description (< 150 tokens)
- Cap maximum description length at 160 tokens
- Remove irrelevant metadata (product codes, dates, etc.)

### 2. Dataset Balancing
The raw data was heavily skewed toward cheap items. The curation process:
- Created price buckets ($1-$999)
- Sampled strategically to balance price distribution
- Reduced over-representation of Automotive category
- Maintained diverse product categories

### 3. Text Preprocessing
- Scrubbed unnecessary characters and formatting
- Removed product codes and SKUs
- Cleaned whitespace and punctuation
- Truncated to optimal token length

## Baseline Models

Multiple baseline approaches were implemented to establish performance benchmarks:

### 1. **Random Pricer**
Random price prediction between $1-$1000.

### 2. **Constant Pricer**
Always predicts the training set average price (~$60).

### 3. **Linear Regression on Features**
Uses extracted features:
- Item weight
- Best sellers rank
- Description length
- Brand (top electronics brands)

### 4. **Bag of Words + Linear Regression**
- CountVectorizer with 1000 features
- Linear regression on word frequencies

### 5. **Word2Vec + Linear Regression**
- 400-dimensional Word2Vec embeddings
- Document vectors averaged from word vectors
- Linear regression on embeddings

### 6. **Word2Vec + Support Vector Regression (SVR)**
- Same Word2Vec embeddings
- LinearSVR for non-linear relationships

### 7. **Word2Vec + Random Forest**
- Same Word2Vec embeddings
- Random Forest with 100 estimators
- Best performing baseline model

## Evaluation Metrics

Models are evaluated using:

1. **Mean Absolute Error (MAE):** Average dollar difference between prediction and truth
2. **Root Mean Squared Logarithmic Error (RMSLE):** Accounts for relative differences
3. **Hit Rate:** Percentage of predictions within 20% or $40 of actual price
4. **Color-coded Results:**
   - 🟢 Green: Error < $40 or < 20% of truth
   - 🟠 Orange: Error < $80 or < 40% of truth
   - 🔴 Red: Error > $80 and > 40% of truth


## **Environment Variables**
Create a .env file with:
- HF_TOKEN=your_huggingface_token
- OPENAI_API_KEY=your_openai_key
- ANTHROPIC_API_KEY=your_anthropic_key

## **Results Summary**
The baseline models establish a performance hierarchy:

- Random Forest performs best among traditional approaches
- Word embeddings significantly outperform bag-of-words
- Feature engineering alone provides limited accuracy
- Deep learning and fine-tuned LLMs show substantial improvements

## **Future Work**

- Incorporate product images for multimodal pricing
- Experiment with category-specific models
- Test larger language models
- Add real-time price prediction API
- Expand to international markets and currencies


## **Acknowledgments**

- Dataset: McAuley-Lab for the Amazon Reviews 2023 dataset
- Base Model: Meta's Llama 3.1-8B for tokenization
- Inspiration: Real-world e-commerce pricing challenges
