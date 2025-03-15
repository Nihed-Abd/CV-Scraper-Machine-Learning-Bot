import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(data_path):
    """
    Preprocess the data from a CSV file for ML models
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        X_train, X_test, y_train, y_test: Split data for training and testing
    """
    logger.info(f"Preprocessing data from {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Feature engineering
    
    # 1. Create a target variable - for this example we'll predict whether the candidate has 
    # more than 3 years of experience (a common cutoff for junior to mid-level roles)
    df['is_experienced'] = (df['years_of_experience'] > 3).astype(int)
    
    # 2. Extract features from skills
    skills_list = []
    for skills in df['skills'].str.split(', '):
        if isinstance(skills, list):
            skills_list.extend(skills)
    
    # Get the most common skills
    top_skills = pd.Series(skills_list).value_counts().nlargest(20).index.tolist()
    
    # Create binary features for each top skill
    for skill in top_skills:
        df[f'has_{skill.lower().replace(" ", "_")}'] = df['skills'].str.contains(skill, case=False).astype(int)
    
    # 3. Extract features from education
    # Convert degree to ordinal feature
    degree_mapping = {
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }
    df['degree_level'] = df['degree'].map(degree_mapping).fillna(0).astype(int)
    
    # 4. One-hot encode the field of study
    if len(df) > 1:  # Only if we have enough samples
        field_dummies = pd.get_dummies(df['field'], prefix='field')
        df = pd.concat([df, field_dummies], axis=1)
    
    # 5. Select features for the model
    feature_cols = [col for col in df.columns if col.startswith('has_') or col.startswith('field_')]
    feature_cols.extend(['degree_level', 'graduation_year', 'years_of_experience'])
    
    # Handle missing values
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        elif df[col].dtype == np.float64 or df[col].dtype == np.int64:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('')
    
    # Select only columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # If no features remain, add a dummy feature
    if not feature_cols:
        df['dummy_feature'] = 1
        feature_cols = ['dummy_feature']
    
    # Split the data
    X = df[feature_cols]
    y = df['is_experienced']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    logger.info(f"Data preprocessing complete. Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_knn_model(X_train, y_train):
    """Train a K-Nearest Neighbors model"""
    logger.info("Training KNN model")
    
    # Create a pipeline with preprocessing and KNN model
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    
    # Train the model
    knn_pipeline.fit(X_train, y_train)
    
    return knn_pipeline

def train_svm_model(X_train, y_train):
    """Train a Support Vector Machine model"""
    logger.info("Training SVM model")
    
    # Create a pipeline with preprocessing and SVM model
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True))
    ])
    
    # Train the model
    svm_pipeline.fit(X_train, y_train)
    
    return svm_pipeline

def train_adaboost_model(X_train, y_train):
    """Train an AdaBoost model"""
    logger.info("Training AdaBoost model")
    
    # Create base classifier
    base_clf = DecisionTreeClassifier(max_depth=1)
    
    # Create a pipeline with preprocessing and AdaBoost model
    ada_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ada', AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, learning_rate=1.0))
    ])
    
    # Train the model
    ada_pipeline.fit(X_train, y_train)
    
    return ada_pipeline

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return performance metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Create a dictionary of metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

def plot_model_comparison(metrics_list):
    """Plot a comparison of model metrics"""
    # Create directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    model_names = [m['model_name'] for m in metrics_list]
    accuracies = [m['accuracy'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1_scores = [m['f1_score'] for m in metrics_list]
    
    # Set up the plot
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    
    # Plot bars
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy')
    ax.bar(x - 0.5*width, precisions, width, label='Precision')
    ax.bar(x + 0.5*width, recalls, width, label='Recall')
    ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score')
    
    # Customize plot
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plt_path)
    
    return plt_path

def train_and_evaluate_models(data_path):
    """
    Train and evaluate KNN, SVM and AdaBoost models on the CV data
    
    Args:
        data_path: Path to the CSV file with CV data
        
    Returns:
        evaluation_text: Summary of model evaluations
    """
    logger.info(f"Starting model training and evaluation on data from {data_path}")
    
    try:
        # Preprocess the data
        X_train, X_test, y_train, y_test, feature_cols = preprocess_data(data_path)
        
        # Check if we have enough data for meaningful modeling
        if len(X_train) < 10 or len(set(y_train)) < 2:
            logger.warning("Not enough data for meaningful modeling")
            return "Not enough diversity in the data for meaningful modeling. Please collect more varied CV data."
        
        # Train the models
        knn_model = train_knn_model(X_train, y_train)
        svm_model = train_svm_model(X_train, y_train)
        ada_model = train_adaboost_model(X_train, y_train)
        
        # Evaluate the models
        knn_metrics = evaluate_model(knn_model, X_test, y_test, "KNN")
        svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")
        ada_metrics = evaluate_model(ada_model, X_test, y_test, "AdaBoost")
        
        # Plot model comparison
        metrics_list = [knn_metrics, svm_metrics, ada_metrics]
        plot_path = plot_model_comparison(metrics_list)
        
        # Find the best model
        best_model = max(metrics_list, key=lambda x: x['f1_score'])
        
        # Create a feature importance analysis for AdaBoost (if it's the best model)
        feature_importance_text = ""
        if best_model['model_name'] == "AdaBoost":
            feature_importances = ada_model.named_steps['ada'].feature_importances_
            feature_names = feature_cols
            
            if len(feature_importances) == len(feature_names):
                importance_pairs = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
                
                feature_importance_text = "\n\nFeature Importance Analysis (AdaBoost):\n"
                for name, importance in importance_pairs[:10]:  # Top 10 features
                    feature_importance_text += f"- {name}: {importance:.4f}\n"
        
        # Create the evaluation text
        evaluation_text = "Model Evaluation Results\n"
        evaluation_text += "=======================\n\n"
        
        evaluation_text += "KNN Model:\n"
        evaluation_text += f"- Accuracy: {knn_metrics['accuracy']:.4f}\n"
        evaluation_text += f"- Precision: {knn_metrics['precision']:.4f}\n"
        evaluation_text += f"- Recall: {knn_metrics['recall']:.4f}\n"
        evaluation_text += f"- F1 Score: {knn_metrics['f1_score']:.4f}\n\n"
        
        evaluation_text += "SVM Model:\n"
        evaluation_text += f"- Accuracy: {svm_metrics['accuracy']:.4f}\n"
        evaluation_text += f"- Precision: {svm_metrics['precision']:.4f}\n"
        evaluation_text += f"- Recall: {svm_metrics['recall']:.4f}\n"
        evaluation_text += f"- F1 Score: {svm_metrics['f1_score']:.4f}\n\n"
        
        evaluation_text += "AdaBoost Model:\n"
        evaluation_text += f"- Accuracy: {ada_metrics['accuracy']:.4f}\n"
        evaluation_text += f"- Precision: {ada_metrics['precision']:.4f}\n"
        evaluation_text += f"- Recall: {ada_metrics['recall']:.4f}\n"
        evaluation_text += f"- F1 Score: {ada_metrics['f1_score']:.4f}\n\n"
        
        evaluation_text += f"Best Model: {best_model['model_name']} with F1 Score of {best_model['f1_score']:.4f}"
        
        evaluation_text += feature_importance_text
        
        logger.info("Model training and evaluation complete")
        return evaluation_text
        
    except Exception as e:
        logger.error(f"Error in model training and evaluation: {str(e)}")
        return f"An error occurred during model training and evaluation: {str(e)}"
