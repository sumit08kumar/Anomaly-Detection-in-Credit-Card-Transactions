#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from models.anomaly_detector import AnomalyDetector

def main():
    # Load data
    print("Loading credit card transaction data...")
    df = pd.read_csv('data/creditcard.csv')
    print(f"Loaded {len(df)} transactions")
    print(f"Features: {list(df.columns)}")
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Train models (excluding the Class column for unsupervised learning)
    training_data = df.drop('Class', axis=1) if 'Class' in df.columns else df
    detector.train_models(training_data)
    
    # Evaluate models if labels are available
    if 'Class' in df.columns:
        print("\nEvaluating models...")
        for model_name in detector.trained_models.keys():
            print(f"\n--- {model_name.upper()} ---")
            results = detector.evaluate_model(df, model_name)
            if results:
                print(f"AUC Score: {results['auc_score']:.4f}" if results['auc_score'] else "AUC Score: N/A")
                print("Classification Report:")
                print(f"Precision (Fraud): {results['classification_report']['1']['precision']:.4f}")
                print(f"Recall (Fraud): {results['classification_report']['1']['recall']:.4f}")
                print(f"F1-Score (Fraud): {results['classification_report']['1']['f1-score']:.4f}")
    
    # Save models
    detector.save_models('models/trained_models.pkl')
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main()

