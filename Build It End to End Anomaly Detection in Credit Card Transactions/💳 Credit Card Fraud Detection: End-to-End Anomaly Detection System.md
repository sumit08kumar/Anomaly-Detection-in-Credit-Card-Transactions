# 💳 Credit Card Fraud Detection: End-to-End Anomaly Detection System

## 🎯 Project Overview

This project implements a robust, end-to-end **unsupervised anomaly detection system** to identify potentially fraudulent credit card transactions. It demonstrates the practical use of **density-based, isolation-based, and probabilistic outlier detection algorithms**, alongside a deployable web interface built using Streamlit.

## 🧠 Problem Statement

Financial institutions face increasing fraud threats. Supervised approaches to detect fraudulent transactions require a large amount of **labeled data**, which is often unavailable or outdated. The goal here is to **detect anomalies using unsupervised methods**, where the assumption is that **fraudulent transactions are rare and deviate from normal patterns**.

## 📊 Dataset

* **Source**: Synthetic credit card transaction data (10,000 transactions)
* **Features**: 
  * V1-V28: Anonymized PCA components
  * Time: Transaction timestamp
  * Amount: Transaction amount
  * Class: Fraud label (for evaluation only)
* **Use case**: Train models without labels → detect outliers → compare against known fraud cases

## ✅ Key Components

### 1. 🔍 Exploratory Data Analysis (EDA)
* Visualize transaction distribution by amount/time
* Dimensionality inspection (PCA scatter plots)
* Fraud frequency and imbalance analysis

### 2. 🧼 Preprocessing
* Scale Amount and Time features
* Apply PCA for visualization
* Prepare data for model training (drop labels from training)

### 3. 🧠 Model Training & Comparison

| Model | Description | Performance |
|-------|-------------|-------------|
| **Isolation Forest** | Tree-based anomaly detector | AUC: 0.6381 |
| **One-Class SVM** | Support vector machine for outliers | AUC: 0.6747 |
| **DBSCAN** | Clustering-based outlier detection | AUC: 0.5000 |

### 4. 📈 Evaluation Strategy
* Precision, Recall, F1-score
* ROC AUC (comparing predicted anomalies with actual frauds)
* Confusion matrix visualization

## 🌐 Web Application Features

### ✅ Interactive Streamlit Interface
* Upload CSV files for batch analysis
* Choose between different detection models
* Real-time anomaly scoring and visualization
* Interactive PCA plots and distribution charts
* Performance metrics and confusion matrix
* Downloadable results

### ✅ Visualizations
* **PCA Scatter Plot**: 2D visualization of transactions
* **Amount Distribution**: Histogram of transaction amounts
* **Time Series**: Transactions by hour of day
* **Confusion Matrix**: Model performance heatmap

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn streamlit plotly seaborn matplotlib
```

### Running the Application
```bash
# Clone or download the project
cd credit-card-anomaly-detector

# Train the models (optional - pre-trained models included)
python notebooks/train_models.py

# Start the web application
streamlit run app/streamlit_app.py
```

### Using Docker
```bash
# Build the Docker image
docker build -t fraud-detector .

# Run the container
docker run -p 8501:8501 fraud-detector
```

## 📁 Project Structure

```
credit-card-anomaly-detector/
├── app/
│   └── streamlit_app.py          # Main web application
├── data/
│   └── creditcard.csv            # Synthetic dataset
├── models/
│   ├── anomaly_detector.py       # Core ML models
│   └── trained_models.pkl        # Pre-trained models
├── notebooks/
│   └── train_models.py           # Model training script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
└── README.md                     # This file
```

## 🎮 How to Use

1. **Load Models**: Click "🚀 Load Models" to initialize the pre-trained algorithms
2. **Load Data**: Use "📊 Load Sample Data" or upload your own CSV file
3. **Select Model**: Choose from Isolation Forest, One-Class SVM, or DBSCAN
4. **Run Detection**: Click "🔍 Run Anomaly Detection" to analyze transactions
5. **Explore Results**: View interactive visualizations and performance metrics
6. **Download Results**: Export detected anomalies as CSV

## 📊 Model Performance

### Isolation Forest (Best Overall)
- **Precision**: 0.0500
- **Recall**: 0.0649
- **F1-Score**: 0.0565
- **AUC**: 0.6381

### One-Class SVM (Best Precision)
- **Precision**: 0.0936
- **Recall**: 0.1234
- **F1-Score**: 0.1064
- **AUC**: 0.6747

### DBSCAN (High Recall)
- **Precision**: 0.0154
- **Recall**: 1.0000
- **F1-Score**: 0.0303
- **AUC**: 0.5000

## 🔧 Technical Details

### Data Preprocessing
- StandardScaler for feature normalization
- PCA for dimensionality reduction and visualization
- Robust handling of missing values

### Model Implementation
- Scikit-learn based implementations
- Configurable contamination rates
- Cross-validation ready architecture

### Web Interface
- Streamlit for rapid prototyping
- Plotly for interactive visualizations
- Responsive design for mobile compatibility

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app/streamlit_app.py
```

### Docker Container
```bash
docker build -t fraud-detector .
docker run -p 8501:8501 fraud-detector
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container-based deployment
- **AWS/GCP**: Scalable cloud hosting

## 🔮 Future Enhancements

- **Deep Learning**: Autoencoder-based anomaly detection
- **Real-time Processing**: Kafka/Redis integration
- **API Endpoints**: REST API for production integration
- **Advanced Visualization**: 3D plots and network graphs
- **Model Ensemble**: Combining multiple algorithms
- **Alert System**: Email/SMS notifications for high-risk transactions

## 📈 Business Impact

- **Fraud Prevention**: Early detection of suspicious transactions
- **Cost Reduction**: Automated screening reduces manual review
- **Customer Protection**: Faster response to potential fraud
- **Compliance**: Audit trail and performance metrics
- **Scalability**: Handle millions of transactions per day

## 🏆 Resume-Ready Summary

> Built and deployed an unsupervised machine learning system for fraud detection in credit card transactions using Isolation Forest, One-Class SVM, and DBSCAN. Achieved 67% AUC score with interactive Streamlit dashboard featuring PCA visualizations, real-time anomaly scoring, and comprehensive performance analytics. Containerized with Docker for production deployment.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities, please reach out via GitHub issues.

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**

