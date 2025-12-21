# 🔐 Machine Learning for Malicious URL / QR Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A machine learning project for detecting malicious (phishing) URLs using various classification algorithms. This project was developed as part of the WQD7006 Machine Learning course at Universiti Malaya.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [References](#references)
- [License](#license)

---

## 🎯 Overview

In today's digital world, URLs are everywhere - shared across social media platforms like WhatsApp, Instagram, and Messenger. However, attackers exploit these platforms to spread **malicious content** through deceptive URLs and QR codes, leading to:

- 🦠 Malware infections
- 🔓 Data breaches
- 💰 Financial losses

This project develops a machine learning model that can **reliably detect phishing URLs** based on extracted features from the URL structure and domain characteristics.

---

## ❓ Problem Statement

Attackers perform **social media phishing** by utilizing deceptive URLs and QR codes to trick users into clicking or scanning, thereby redirecting them to harmful websites.

### Problem Objectives

1. ✅ Identify URL patterns (lexical, domain-based) indicative of malicious sites
2. ✅ Evaluate and compare the performance of different ML models for malicious URL classification
3. ✅ Develop a ML-based model capable of detecting phishing attempts based on URL patterns reliably

---

## 📊 Dataset

**Dataset:** LegitPhish - A large-scale annotated dataset for URL-based phishing detection

| Metric | Value |
|--------|-------|
| **Total URLs** | 101,219 |
| **Phishing URLs** | 62.9% |
| **Legitimate URLs** | 37.1% |
| **Initial Features** | 18 |

### Data Source
- URLHaus database (malicious URLs)
- Wikipedia & Stack Overflow (legitimate URLs)

### Key Features
| Feature | Description |
|---------|-------------|
| `url_length` | Total number of characters in the URL |
| `has_ip_address` | Binary flag: whether URL contains an IP address |
| `https_flag` | Binary flag: whether URL uses HTTPS |
| `subdomain_count` | Number of subdomains in the URL |
| `url_entropy` | Shannon entropy of the URL string |
| `dot_count` | Number of `.` characters in the URL |
| `token_count` | Number of tokens/words in the URL |
| `query_param_count` | Number of query parameters |
| `tld_length` | Length of the Top-Level Domain |
| `path_length` | Length of the path after the domain |

---

## 🔬 Methodology

### Data Pre-processing
```
Data Collection → Data Cleaning → Feature Extraction → Data Splitting (80/20)
```

### Machine Learning Models
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model for binary classification |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Extreme Gradient Boosting |
| **LightGBM** | Light Gradient Boosting Machine |

### Evaluation Metrics
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ AUC-ROC
- ✅ Confusion Matrix

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab

### Clone the Repository
```bash
git clone https://github.com/Elmer2408/WQD7006_Machine-learning-of-Malicious-URLs-Detection.git
cd WQD7006_Machine-learning-of-Malicious-URLs-Detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib
```

---

## 🚀 Usage

### Option 1: Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Run all cells sequentially
3. View results and visualizations

### Option 2: Local Jupyter Notebook
```bash
jupyter notebook Malicious_URL_Detection_ML.ipynb
```

### Option 3: Load Pre-trained Model
```python
import joblib

# Load the trained model and scaler
model = joblib.load('best_model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

# Predict on new URL features
features = [...]  # Extract features from URL
scaled_features = scaler.transform([features])
prediction = model.predict(scaled_features)

print("Phishing" if prediction[0] == 0 else "Legitimate")
```

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | ~95% | ~95% | ~95% | ~95% | ~98% |
| XGBoost | ~95% | ~95% | ~95% | ~95% | ~98% |
| LightGBM | ~95% | ~95% | ~95% | ~95% | ~98% |
| Logistic Regression | ~90% | ~90% | ~90% | ~90% | ~95% |

*Note: Actual values may vary slightly based on random state*

### Key Findings
1. **Ensemble methods** (Random Forest, XGBoost, LightGBM) outperform Logistic Regression
2. **URL entropy** and **URL length** are the most predictive features
3. All models achieved **>90% accuracy** in detecting phishing URLs

---

## 📁 Project Structure

```
WQD7006_Machine-learning-of-Malicious-URLs-Detection/
│
├── 📓 Malicious_URL_Detection_ML.ipynb    # Main Jupyter notebook
├── 📄 README.md                            # Project documentation
├── 📄 requirements.txt                     # Python dependencies
│
├── 📂 data/
│   ├── url_features_extracted1.csv         # Initial dataset
│   └── urldata_with_features_complete.csv  # Complete dataset with features
│
├── 📂 models/
│   ├── best_model_random_forest.pkl        # Saved best model
│   └── scaler.pkl                          # Saved StandardScaler
│
└── 📂 images/
    └── poster.png                          # Project poster
```

---

## 👥 Authors

| Name | Student ID |
|------|------------|
| YIM WEN JUN | 24201054 |
| RICHIE TEOH | 24088171 |
| ELMER LEE JIA ZHAO | 24082366 |
| ANGELINE TAN JIE LIN | 24084444 |
| MICOLE CHUNG SYN TUNG | 24073625 |

**Institution:** Universiti Malaya  
**Course:** WQD7006 - Machine Learning

---

## 📚 References

1. Potpelwar, R. S., Kulkarni, U. V., & Waghmare, J. M. (2025). *LegitPhish: A large-scale annotated dataset for URL-based phishing detection*. Data in Brief, 63, 111972. https://doi.org/10.1016/j.dib.2025.111972

2. Xuan, C. D., Dinh, H., & Victor, T. (2020). *Malicious URL Detection based on Machine Learning*. International Journal of Advanced Computer Science and Applications, 11(1). https://doi.org/10.14569/ijacsa.2020.0110119

3. Aryan Nandu, Sosa, J., Pant, Y., Panchal, Y., & Sayyad, S. (2024). *Malicious URL Detection Using Machine Learning*. 1–6. https://doi.org/10.1109/asiancon62057.2024.10837752

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Universiti Malaya for providing the educational platform
- LegitPhish dataset creators for the comprehensive URL dataset
- Open-source community for the machine learning libraries

---

<p align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ by WQD7006 Team | © 2025 Universiti Malaya
</p>
