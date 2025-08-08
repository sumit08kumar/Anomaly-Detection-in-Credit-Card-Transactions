# Anomaly-Detection-in-Credit-Card-Transactions

## 📌 Project:

**Anomaly Detection in Credit Card Transactions: An End-to-End Unsupervised ML Pipeline with Web Deployment**

---

## 🧠 Project Summary:

This project implements a robust, end-to-end **unsupervised anomaly detection system** to identify potentially fraudulent credit card transactions in a dataset without explicit fraud labels. It demonstrates the practical use of **density-based, isolation-based, and probabilistic outlier detection algorithms**, alongside a deployable web interface built using Streamlit and Docker.

---

## 🎯 Problem Statement:

Financial institutions face increasing fraud threats. Supervised approaches to detect fraudulent transactions require a large amount of **labeled data**, which is often unavailable or outdated. The goal here is to **detect anomalies using unsupervised methods**, where the assumption is that **fraudulent transactions are rare and deviate from normal patterns**.

---

## 📊 Dataset:

* **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description**:

  * 284,807 transactions
  * 492 fraud cases (only for validation, not training)
  * Features are anonymized via PCA: `V1` to `V28`, `Amount`, `Time`
* **Use case**: Train models without labels → detect outliers → compare against known fraud cases (optional)

---

## ✅ Key Components:

### 1. 🔍 Exploratory Data Analysis (EDA)

* Visualize transaction distribution by amount/time
* Dimensionality inspection (PCA scatter plots)
* Fraud frequency and imbalance analysis (used only for final validation)

---

### 2. 🧼 Preprocessing

* Scale `Amount` and `Time` features
* Apply additional dimensionality reduction (`PCA`, `t-SNE`) for visualization
* Prepare data for model training (drop labels from training)

---

### 3. 🧠 Model Training & Comparison

| Model                | Description                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Isolation Forest** | Tree-based anomaly detector based on isolation                                                                 |
| **One-Class SVM**    | Classifies data as inliers vs outliers                                                                         |
| **DBSCAN**           | Clustering-based outlier detection                                                                             |
| **Autoencoder**      | Reconstruct normal transactions; detect high reconstruction error as anomalies (optional deep learning add-on) |

---

### 4. 📈 Evaluation Strategy

Even though labels are not used for training, they are used **post-hoc** for evaluation:

* Precision, Recall, F1-score
* ROC AUC (comparing predicted anomalies with actual frauds)
* Confusion matrix (True Positives = correctly flagged frauds)

---

## 🌐 Deployment

### ✅ Web Interface (Streamlit)

* Upload a batch of transactions (CSV)
* Choose model (Isolation Forest, One-Class SVM, etc.)
* See flagged anomalies + probability scores
* Optionally visualize transactions on 2D plot (PCA/t-SNE)

### ✅ Dockerization

* Fully containerized using Docker
* Easy deployment to:

  * **Localhost**
  * **Render / GCP Cloud Run / AWS EC2 / Streamlit Cloud**

---

## 📁 Suggested Folder Structure

```
credit-card-anomaly-detector/
├── app/
│   └── streamlit_app.py
├── data/
│   └── creditcard.csv
├── models/
│   └── isolation_forest.pkl
│   └── scaler.pkl
├── notebooks/
│   └── EDA.ipynb
│   └── Anomaly_Modeling.ipynb
├── requirements.txt
├── Dockerfile
├── README.md
```

---

## ✨ Resume-Ready Bullet Point

> ✅ Built and deployed an unsupervised machine learning system for fraud detection in credit card transactions using Isolation Forest, One-Class SVM, and DBSCAN. Visualized transaction embeddings using PCA/t-SNE, evaluated detection performance against labeled fraud instances, and deployed an interactive Streamlit app via Docker for real-time anomaly scoring.

---

## 🧠 Bonus Ideas (Advanced)

* Add **Autoencoder-based anomaly detection** (deep learning)
* Build a **REST API using FastAPI** for production-style scoring
* Integrate **email/SMS alerting** on anomaly detection (mocked)

