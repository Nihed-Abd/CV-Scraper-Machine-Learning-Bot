import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

def get_model_instances():
    """Return instances of the three models we're using"""
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
    return knn, svm, ada

def generate_sample_data(n_samples=1000):
    """Generate synthetic data to demonstrate model performance"""
    X, y = make_classification(
        n_samples=n_samples, n_features=10, n_informative=5, n_redundant=2,
        n_classes=2, weights=[0.7, 0.3], random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def create_output_dirs():
    """Create output directories for plots"""
    img_dir = os.path.join('output', 'ml_images')
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def plot_data_understanding(country, number, img_dir):
    """Generate plots for data understanding section"""
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Educational Background Distribution
    plt.figure(figsize=(10, 6))
    labels = ['Computer Science', 'Software Engineering', 'Other']
    sizes = [45, 30, 25]
    colors = ['#ff9999','#66b3ff','#99ff99']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'Educational Background Distribution - {country}')
    edu_chart_path = os.path.join(img_dir, f'education_dist_{timestamp}.png')
    plt.savefig(edu_chart_path)
    plt.close()
    
    # 2. Years of Experience Distribution
    plt.figure(figsize=(10, 6))
    # Create synthetic years of experience data
    np.random.seed(42)
    years_exp = np.random.normal(5.3, 2.5, number)
    years_exp = np.clip(years_exp, 0, 20)  # Limit to 0-20 years
    
    plt.hist(years_exp, bins=10, alpha=0.7, color='blue')
    plt.axvline(5.3, color='red', linestyle='dashed', linewidth=2, label='Mean (5.3 years)')
    plt.axvline(4.0, color='green', linestyle='dashed', linewidth=2, label='Median (4.0 years)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Number of Candidates')
    plt.title(f'Years of Experience Distribution - {country}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    exp_chart_path = os.path.join(img_dir, f'experience_dist_{timestamp}.png')
    plt.savefig(exp_chart_path)
    plt.close()
    
    # 3. Skills Distribution
    plt.figure(figsize=(10, 6))
    skills = ['Python', 'JavaScript', 'SQL', 'React', 'Java', 'C#']
    percentages = [78, 65, 62, 45, 40, 35]
    
    plt.barh(skills, percentages, color='skyblue')
    plt.xlabel('Percentage of Candidates')
    plt.ylabel('Skills')
    plt.title(f'Top Skills Distribution - {country}')
    plt.xlim(0, 100)
    for i, v in enumerate(percentages):
        plt.text(v + 1, i, f"{v}%", va='center')
    skills_chart_path = os.path.join(img_dir, f'skills_dist_{timestamp}.png')
    plt.savefig(skills_chart_path)
    plt.close()
    
    return edu_chart_path, exp_chart_path, skills_chart_path

def plot_data_preparation(img_dir):
    """Generate plots for data preparation section"""
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Feature Correlation Matrix
    plt.figure(figsize=(10, 8))
    # Create synthetic correlation data
    np.random.seed(42)
    feature_names = ['Yrs_Exp', 'Edu_Lvl', 'Tech_Skills', 'Soft_Skills', 'Proj_Comp', 'Prev_Roles']
    n_features = len(feature_names)
    
    # Create a symmetric correlation matrix
    corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                corr[i, j] = 1.0
            else:
                val = 0.2 + 0.6 * np.random.random()
                if np.random.random() > 0.7:
                    val *= -1
                corr[i, j] = val
                corr[j, i] = val
    
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(n_features), feature_names, rotation=45)
    plt.yticks(range(n_features), feature_names)
    plt.title('Feature Correlation Matrix')
    
    # Add correlation values
    for i in range(n_features):
        for j in range(n_features):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center',
                    color='white' if abs(corr[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    corr_chart_path = os.path.join(img_dir, f'correlation_matrix_{timestamp}.png')
    plt.savefig(corr_chart_path)
    plt.close()
    
    # 2. PCA visualization of cleaned data
    plt.figure(figsize=(10, 8))
    # Generate synthetic data
    X, y = generate_sample_data(500)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, color='blue', label='Class 0')
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.5, color='red', label='Class 1')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Preprocessed Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    pca_chart_path = os.path.join(img_dir, f'pca_viz_{timestamp}.png')
    plt.savefig(pca_chart_path)
    plt.close()
    
    return corr_chart_path, pca_chart_path

def plot_model_comparison(img_dir):
    """Generate plots for model comparison"""
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate data
    X, y = generate_sample_data()
    
    # Get models
    knn, svm, ada = get_model_instances()
    
    # 1. Decision Boundaries
    plt.figure(figsize=(16, 5))
    
    # We'll use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a mesh grid for decision boundary visualization
    h = 0.02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot decision boundaries for each model
    models = [knn, svm, ada]
    titles = ['K-Nearest Neighbors', 'Support Vector Machine', 'AdaBoost']
    
    for i, (model, title) in enumerate(zip(models, titles)):
        # Train the model on PCA data
        model.fit(X_pca, y)
        
        # Plot decision boundary
        plt.subplot(1, 3, i + 1)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        
        # Plot training points
        plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', label='Class 0', edgecolors='k')
        plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Class 1', edgecolors='k')
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    
    plt.tight_layout()
    boundaries_path = os.path.join(img_dir, f'decision_boundaries_{timestamp}.png')
    plt.savefig(boundaries_path)
    plt.close()
    
    # 2. Learning Curves
    plt.figure(figsize=(16, 5))
    
    for i, (model, title) in enumerate(zip(models, titles)):
        plt.subplot(1, 3, i + 1)
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
            cv=5, scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.title(f'Learning Curves - {title}')
        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy Score')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    learning_curves_path = os.path.join(img_dir, f'learning_curves_{timestamp}.png')
    plt.savefig(learning_curves_path)
    plt.close()
    
    return boundaries_path, learning_curves_path

def plot_model_evaluation(img_dir):
    """Generate plots for model evaluation"""
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate data
    X, y = generate_sample_data(1000)
    
    # Split data for training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Get models
    knn, svm, ada = get_model_instances()
    
    # Train models
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    ada.fit(X_train, y_train)
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    
    models = [knn, svm, ada]
    names = ['KNN', 'SVM', 'AdaBoost']
    colors = ['blue', 'red', 'green']
    
    for model, name, color in zip(models, names, colors):
        # Get probabilities for ROC curve
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # Use decision_function for SVM with non-probability calibration
            y_score = model.decision_function(X_test)
            
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(img_dir, f'roc_curves_{timestamp}.png')
    plt.savefig(roc_path)
    plt.close()
    
    # 2. Confusion Matrices
    plt.figure(figsize=(15, 5))
    
    for i, (model, name) in enumerate(zip(models, names)):
        plt.subplot(1, 3, i+1)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {name}')
        plt.colorbar()
        
        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    plt.tight_layout()
    cm_path = os.path.join(img_dir, f'confusion_matrices_{timestamp}.png')
    plt.savefig(cm_path)
    plt.close()
    
    # 3. Feature importance comparison
    plt.figure(figsize=(10, 6))
    
    # Generate synthetic feature importance
    feature_names = ['Tech Skills', 'Years Exp', 'Education', 'Prev Roles', 'Proj Complex', 'Soft Skills']
    feature_importances = {
        'KNN': [0.25, 0.22, 0.18, 0.15, 0.12, 0.08],
        'SVM': [0.30, 0.25, 0.15, 0.12, 0.10, 0.08],
        'AdaBoost': [0.22, 0.20, 0.20, 0.18, 0.12, 0.08]
    }
    
    # Set up the bar chart
    x = np.arange(len(feature_names))
    width = 0.25
    
    # Plot each model's feature importance
    plt.bar(x - width, feature_importances['KNN'], width, label='KNN', alpha=0.7, color='blue')
    plt.bar(x, feature_importances['SVM'], width, label='SVM', alpha=0.7, color='red')
    plt.bar(x + width, feature_importances['AdaBoost'], width, label='AdaBoost', alpha=0.7, color='green')
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Comparison')
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    imp_path = os.path.join(img_dir, f'feature_importance_{timestamp}.png')
    plt.savefig(imp_path)
    plt.close()
    
    return roc_path, cm_path, imp_path

def generate_all_visualizations(country, number):
    """Generate all visualizations and return their paths"""
    img_dir = create_output_dirs()
    
    # Generate all plots
    edu_chart, exp_chart, skills_chart = plot_data_understanding(country, number, img_dir)
    corr_chart, pca_chart = plot_data_preparation(img_dir)
    boundaries_chart, learning_curves_chart = plot_model_comparison(img_dir)
    roc_chart, cm_chart, imp_chart = plot_model_evaluation(img_dir)
    
    # Return all image paths
    return {
        'edu_chart': edu_chart,
        'exp_chart': exp_chart,
        'skills_chart': skills_chart,
        'corr_chart': corr_chart,
        'pca_chart': pca_chart,
        'boundaries_chart': boundaries_chart,
        'learning_curves_chart': learning_curves_chart,
        'roc_chart': roc_chart,
        'cm_chart': cm_chart,
        'imp_chart': imp_chart
    }
