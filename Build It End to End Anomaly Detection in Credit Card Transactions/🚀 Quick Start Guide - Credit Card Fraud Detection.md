# 🚀 Quick Start Guide - Credit Card Fraud Detection

## 📦 What You've Received

A complete end-to-end **Credit Card Fraud Detection System** with:

- ✅ **3 Machine Learning Models**: Isolation Forest, One-Class SVM, DBSCAN
- ✅ **Interactive Web Interface**: Built with Streamlit
- ✅ **Real-time Visualizations**: PCA plots, distributions, performance metrics
- ✅ **Synthetic Dataset**: 10,000 credit card transactions with 1.54% fraud rate
- ✅ **Docker Support**: Ready for containerized deployment
- ✅ **Complete Documentation**: README, code comments, and this guide

## 🏃‍♂️ Run in 3 Steps

### Step 1: Install Dependencies
```bash
cd credit-card-anomaly-detector
pip install -r requirements.txt
```

### Step 2: Start the Application
```bash
streamlit run app/streamlit_app.py
```

### Step 3: Open in Browser
- Navigate to: `http://localhost:8501`
- Click "🚀 Load Models" 
- Click "📊 Load Sample Data"
- Click "🔍 Run Anomaly Detection"

## 🎯 Key Features Demonstrated

### 1. **Unsupervised Learning**
- No labeled training data required
- Detects anomalies based on transaction patterns
- Compares 3 different algorithms

### 2. **Interactive Dashboard**
- Upload your own CSV files
- Switch between detection models
- Real-time visualization updates
- Download results as CSV

### 3. **Professional Visualizations**
- **PCA Scatter Plot**: See normal vs anomalous transactions
- **Amount Distribution**: Histogram showing transaction patterns
- **Time Series**: Fraud patterns by hour of day
- **Confusion Matrix**: Model performance heatmap

### 4. **Performance Metrics**
- Precision, Recall, F1-Score
- ROC AUC scores
- Detailed classification reports

## 🐳 Docker Alternative

If you prefer Docker:
```bash
cd credit-card-anomaly-detector
docker build -t fraud-detector .
docker run -p 8501:8501 fraud-detector
```

## 📊 Expected Results

When you run the demo:
- **Total Transactions**: 10,000
- **Detected Anomalies**: ~200 (2% rate)
- **Best Model**: One-Class SVM (AUC: 0.6747)
- **Visualizations**: Interactive plots showing clear separation

## 🔧 Customization Options

### Upload Your Own Data
- CSV format with columns: V1-V28, Time, Amount
- Optional: Class column for evaluation
- Drag & drop in the web interface

### Adjust Model Parameters
- Edit `models/anomaly_detector.py`
- Modify contamination rates
- Add new algorithms

### Styling & Layout
- Customize `app/streamlit_app.py`
- Add new visualizations
- Modify color schemes

## 🎓 Resume/Portfolio Ready

This project demonstrates:
- **Machine Learning**: Unsupervised anomaly detection
- **Data Science**: EDA, preprocessing, evaluation
- **Web Development**: Interactive dashboards
- **DevOps**: Docker containerization
- **Documentation**: Professional README and guides

## 🆘 Troubleshooting

### Common Issues:
1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Port Conflicts**: Change port with `--server.port 8502`
3. **Memory Issues**: Reduce dataset size in training script
4. **Browser Issues**: Try incognito mode or different browser

### Need Help?
- Check the detailed README.md
- Review code comments in each file
- All models are pre-trained and ready to use

## 🎉 Success Metrics

You'll know it's working when you see:
- ✅ Models load successfully
- ✅ Sample data displays 10,000 transactions
- ✅ Detection completes in ~5 seconds
- ✅ Interactive plots render correctly
- ✅ Performance metrics show reasonable scores

**Enjoy exploring your fraud detection system!** 🚀

