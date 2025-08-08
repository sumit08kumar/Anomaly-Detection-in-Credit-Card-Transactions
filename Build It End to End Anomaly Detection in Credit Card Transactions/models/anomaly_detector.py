import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

class AnomalyDetector:
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.02, random_state=42),
            'one_class_svm': OneClassSVM(gamma='scale', nu=0.02),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.trained_models = {}
        
    def preprocess_data(self, df, fit_scaler=True):
        """Preprocess the data by scaling features"""
        # Select features (exclude Class if present)
        feature_cols = [col for col in df.columns if col != 'Class']
        X = df[feature_cols].copy()
        
        # Scale the features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, feature_cols
    
    def train_models(self, df):
        """Train all anomaly detection models"""
        print("Preprocessing data...")
        X_scaled, feature_cols = self.preprocess_data(df, fit_scaler=True)
        
        # Fit PCA for visualization
        self.pca.fit(X_scaled)
        
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'dbscan':
                # DBSCAN doesn't have a fit method, we'll use fit_predict
                labels = model.fit_predict(X_scaled)
                # Convert DBSCAN labels to anomaly scores (-1 for outliers, others for normal)
                anomaly_scores = np.where(labels == -1, -1, 1)
                self.trained_models[name] = {
                    'model': model,
                    'type': 'dbscan'
                }
            else:
                model.fit(X_scaled)
                self.trained_models[name] = {
                    'model': model,
                    'type': 'standard'
                }
        
        print("Training completed!")
        return self.trained_models
    
    def predict_anomalies(self, df, model_name='isolation_forest'):
        """Predict anomalies using specified model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet!")
            
        X_scaled, _ = self.preprocess_data(df, fit_scaler=False)
        model_info = self.trained_models[model_name]
        model = model_info['model']
        
        if model_info['type'] == 'dbscan':
            # For DBSCAN, we need to predict on new data
            labels = model.fit_predict(X_scaled)
            predictions = np.where(labels == -1, -1, 1)
            scores = np.where(labels == -1, -0.5, 0.5)  # Simple scoring
        else:
            predictions = model.predict(X_scaled)
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_scaled)
            else:
                scores = model.score_samples(X_scaled)
        
        # Convert predictions to binary (1 for normal, 0 for anomaly)
        binary_predictions = np.where(predictions == 1, 0, 1)
        
        return binary_predictions, scores
    
    def get_pca_components(self, df):
        """Get PCA components for visualization"""
        X_scaled, _ = self.preprocess_data(df, fit_scaler=False)
        return self.pca.transform(X_scaled)
    
    def evaluate_model(self, df, model_name='isolation_forest'):
        """Evaluate model performance if true labels are available"""
        if 'Class' not in df.columns:
            print("No true labels available for evaluation")
            return None
            
        predictions, scores = self.predict_anomalies(df, model_name)
        true_labels = df['Class'].values
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        try:
            auc_score = roc_auc_score(true_labels, -scores)  # Negative because lower scores indicate anomalies
        except:
            auc_score = None
            
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc_score
        }
    
    def save_models(self, filepath):
        """Save trained models and preprocessors"""
        save_data = {
            'trained_models': self.trained_models,
            'scaler': self.scaler,
            'pca': self.pca
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models and preprocessors"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        self.trained_models = save_data['trained_models']
        self.scaler = save_data['scaler']
        self.pca = save_data['pca']
        print(f"Models loaded from {filepath}")

