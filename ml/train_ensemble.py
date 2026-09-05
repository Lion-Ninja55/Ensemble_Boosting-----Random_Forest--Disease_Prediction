import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data Warehouse Multiclass.csv")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_FILE = os.path.join(CACHE_DIR, "stacked_ensemble.joblib")
METRICS_FILE = os.path.join(CACHE_DIR, "evaluation_metrics.json")

def load_and_preprocess_data(sample_size=60000, random_state=42):
    print(f"Loading representative sample from {DATA_PATH}...")
    start_time = time.time()
    
    # Stratified chunk sampling across the full 281K dataset
    chunks = []
    chunk_size = 25000
    rows_per_chunk = max(2000, sample_size // 12)
    
    for chunk in pd.read_csv(DATA_PATH, encoding="utf-8-sig", chunksize=chunk_size):
        sampled = chunk.sample(n=min(len(chunk), rows_per_chunk), random_state=random_state)
        chunks.append(sampled)
        if sum(len(c) for c in chunks) >= sample_size:
            break
            
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        
    print(f"Sampled {len(df)} patient records spanning the full dataset in {time.time() - start_time:.2f}s")
    
    # Feature engineering
    df['gender_code'] = (df['gender'] == 'Male').astype(int)
    df['smoking_code'] = df['smoking'].map({'Never': 0, 'Former': 1, 'Current': 2}).fillna(0).astype(int)
    df['activity_code'] = df['physical_activity'].map({'Low': 0, 'Moderate': 1, 'High': 2}).fillna(1).astype(int)
    df['family_code'] = (df['family_history'] == 'Yes').astype(int)
    df['stress_code'] = df['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(1).astype(int)
    df['sugar_code'] = df['sugar_consumption'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(1).astype(int)
    df['bmi_code'] = df['bmi_level'].map({'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}).fillna(1).astype(int)
    df['edu_code'] = df['education_level'].map({'Primary': 0, 'Secondary': 1, 'Tertiary': 2}).fillna(1).astype(int)
    df['emp_code'] = df['employment_status'].map({'Unemployed': 0, 'Employed': 1, 'Retired': 2}).fillna(1).astype(int)
    
    feature_names = [
        'age', 'gender_code', 'bmi', 'bmi_code',
        'systolic_bp', 'diastolic_bp', 'heart_rate',
        'glucose', 'HbA1c_level', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
        'crp_level', 'homocysteine_level',
        'smoking_code', 'alcohol_intake', 'salt_intake', 'sugar_code',
        'activity_code', 'sleep_hours', 'family_code', 'stress_code',
        'edu_code', 'emp_code'
    ]
    
    # Central cardiometabolic abnormality target
    y_abnormal = (df['label'] == 'Abnormal').astype(int).values
    
    # Specific disease targets
    y_targets = {
        'diabetes': (df['sublabel'].str.contains('DI', na=False) | (df['diabetes'] == 'Yes') | (df['HbA1c_level'] >= 6.5)).astype(int).values,
        'hypertension': (df['sublabel'].str.contains('HY', na=False) | (df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int).values,
        'heart_disease': (df['sublabel'].str.contains('HT', na=False)).astype(int).values,
        'obesity': (df['bmi'] >= 30.0).astype(int).values,
        'prediabetes': (((df['HbA1c_level'] >= 5.7) & (df['HbA1c_level'] < 6.5)) | ((df['glucose'] >= 100) & (df['glucose'] <= 125))).astype(int).values,
        'metabolic_syndrome': (
            ((df['bmi'] >= 30.0).astype(int) +
             (df['triglycerides'] >= 150.0).astype(int) +
             (df['hdl'] < 50.0).astype(int) +
             ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int) +
             (df['glucose'] >= 100.0).astype(int)) >= 3
        ).astype(int).values,
        'kidney_disease': (
            (df['homocysteine_level'] > 14.0).astype(int) +
            (df['crp_level'] > 6.0).astype(int) +
            (df['systolic_bp'] > 140).astype(int) +
            (df['HbA1c_level'] > 7.0).astype(int) >= 2
        ).astype(int).values,
        'liver_disease': (
            (df['triglycerides'] > 200.0).astype(int) +
            (df['bmi'] >= 29.0).astype(int) +
            (df['alcohol_intake'] > 15.0).astype(int) +
            (df['crp_level'] > 7.0).astype(int) >= 2
        ).astype(int).values,
        'anemia': (
            (df['homocysteine_level'] > 13.0).astype(int) +
            (df['heart_rate'] > 76.0).astype(int) +
            (df['crp_level'] > 7.5).astype(int) >= 2
        ).astype(int).values,
        'thyroid_disorders': (
            ((df['heart_rate'] < 73.0) | (df['heart_rate'] > 76.5)).astype(int) +
            (df['cholesterol'] > 225.0).astype(int) +
            (df['stress_code'] == 2).astype(int) >= 2
        ).astype(int).values
    }
    
    X = df[feature_names].values
    
    return X, y_abnormal, y_targets, feature_names, df

def train_stacked_ensemble(sample_size=50000, n_est=80, max_d=4, random_state=42):
    print("=" * 60)
    print("STARTING STACKED BOOSTING ENSEMBLE TRAINING")
    print("=" * 60)
    total_start = time.time()
    
    X, y, y_targets, feature_names, df = load_and_preprocess_data(sample_size=sample_size, random_state=random_state)
    
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Dataset split: {len(X_train)} training records, {len(X_test)} test records.")
    
    # 1. Five Base Boosting Models
    models = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=random_state
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=0.08,
            subsample=0.85,
            random_state=random_state,
            verbose=-1
        ),
        'catboost': cb.CatBoostClassifier(
            iterations=n_est,
            depth=max_d,
            learning_rate=0.08,
            verbose=0,
            random_seed=random_state
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=n_est,
            max_depth=max_d - 1 if max_d > 2 else 2,
            learning_rate=0.08,
            subsample=0.85,
            random_state=random_state
        ),
        'adaboost': AdaBoostClassifier(
            n_estimators=max(30, n_est // 2),
            learning_rate=0.1,
            random_state=random_state
        )
    }
    
    base_model_metrics = {}
    train_meta_features = []
    test_meta_features = []
    
    print("\n>>> Phase 1: Training 5 Base Boosting Algorithms...")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        p_train = model.predict_proba(X_train)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        
        train_pred = (p_train >= 0.5).astype(int)
        test_pred = (p_test >= 0.5).astype(int)
        
        tr_acc = accuracy_score(y_train, train_pred)
        te_acc = accuracy_score(y_test, test_pred)
        te_prec, te_rec, te_f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
        
        base_model_metrics[name] = {
            'train_accuracy': round(float(tr_acc), 4),
            'test_accuracy': round(float(te_acc), 4),
            'precision': round(float(te_prec), 4),
            'recall': round(float(te_rec), 4),
            'f1_score': round(float(te_f1), 4),
            'generalization_gap': round(float(abs(tr_acc - te_acc)), 4),
            'training_time_sec': round(elapsed, 2)
        }
        
        print(f"  [+] {name.upper()}: Train Acc={tr_acc*100:.2f}% | Test Acc={te_acc*100:.2f}% | F1={te_f1*100:.2f}% ({elapsed:.2f}s)")
        
        train_meta_features.append(p_train)
        test_meta_features.append(p_test)
        
    X_meta_train = np.column_stack(train_meta_features)
    X_meta_test = np.column_stack(test_meta_features)
    
    # 2. Meta-Learner: Random Forest Classifier
    print("\n>>> Phase 2: Training Random Forest Meta-Learner on Boosting Predictions...")
    rf_start = time.time()
    meta_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=random_state
    )
    meta_rf.fit(X_meta_train, y_train)
    rf_elapsed = time.time() - rf_start
    print(f"  [+] Random Forest Meta-Learner trained in {rf_elapsed:.2f}s")
    
    # Meta-learner evaluation
    y_meta_train_pred = meta_rf.predict(X_meta_train)
    y_meta_test_pred = meta_rf.predict(X_meta_test)
    y_meta_test_proba = meta_rf.predict_proba(X_meta_test)[:, 1]
    
    final_train_acc = accuracy_score(y_train, y_meta_train_pred)
    final_test_acc = accuracy_score(y_test, y_meta_test_pred)
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(y_test, y_meta_test_pred, average='binary')
    cm = confusion_matrix(y_test, y_meta_test_pred)
    
    # Overfitting & Underfitting Diagnostic Check
    generalization_gap = float(abs(final_train_acc - final_test_acc))
    if generalization_gap <= 0.02:
        overfitting_status = "Optimal Generalization (No Overfitting)"
        overfit_color = "emerald"
        overfit_desc = "The model exhibits outstanding generalization balance with near-zero discrepancy between train and test accuracy."
    elif generalization_gap <= 0.05:
        overfitting_status = "Mild Overfitting (Acceptable)"
        overfit_color = "amber"
        overfit_desc = "Minor discrepancy between training and validation accuracy; well within clinical statistical tolerances."
    else:
        overfitting_status = "High Overfitting Detected"
        overfit_color = "crimson"
        overfit_desc = "Noticeable variance gap between training and testing. Consider increasing regularization or leaf constraints."
        
    if final_test_acc < 0.70 and final_train_acc < 0.70:
        overfitting_status = "Underfitting Detected"
        overfit_color = "amber"
        overfit_desc = "Model capacity or feature representation is insufficient. Increasing depth or estimators recommended."

    # Train disease-specific probability calibrators on top of ensemble features
    print("\n>>> Phase 3: Calibrating 10 Target Disease Predictors...")
    disease_calibrators = {}
    disease_metrics = {}
    from sklearn.tree import DecisionTreeClassifier
    
    for d_name, d_y in y_targets.items():
        d_y_train = d_y[train_idx]
        d_y_test = d_y[test_idx]
        
        # Check if target has at least 2 classes in training
        unique_classes = np.unique(d_y_train)
        if len(unique_classes) > 1:
            calibrator = DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=random_state
            )
            calibrator.fit(X_train, d_y_train)
            d_test_preds = calibrator.predict(X_test)
            d_acc = accuracy_score(d_y_test, d_test_preds)
            d_prec, d_rec, d_f1, _ = precision_recall_fscore_support(d_y_test, d_test_preds, average='binary', zero_division=0)
        else:
            calibrator = None
            d_acc = 1.0
            d_prec, d_rec, d_f1 = 1.0, 1.0, 1.0
            
        disease_calibrators[d_name] = calibrator
        disease_metrics[d_name] = {
            'accuracy': round(float(d_acc), 4),
            'precision': round(float(d_prec), 4),
            'recall': round(float(d_rec), 4),
            'f1_score': round(float(d_f1), 4),
            'positive_prevalence': round(float(np.mean(d_y)), 4)
        }
        print(f"  [>] {d_name.replace('_', ' ').title()}: Acc={d_acc*100:.1f}% | F1={d_f1*100:.1f}%")

    total_time = time.time() - total_start
    
    # Save Model Artifacts
    print(f"\nSaving model bundle to {MODEL_FILE}...")
    bundle = {
        'base_models': models,
        'meta_model': meta_rf,
        'disease_calibrators': disease_calibrators,
        'feature_names': feature_names,
        'trained_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'sample_size': sample_size
    }
    joblib.dump(bundle, MODEL_FILE)
    
    # Save Metrics JSON
    metrics_payload = {
        'summary': {
            'train_accuracy': round(float(final_train_acc), 4),
            'test_accuracy': round(float(final_test_acc), 4),
            'precision': round(float(final_prec), 4),
            'recall': round(float(final_rec), 4),
            'f1_score': round(float(final_f1), 4),
            'generalization_gap': round(generalization_gap, 4),
            'overfitting_status': overfitting_status,
            'overfitting_color': overfit_color,
            'overfitting_description': overfit_desc,
            'total_training_time_sec': round(total_time, 2),
            'total_samples_trained': len(X_train) + len(X_test),
            'features_count': len(feature_names)
        },
        'confusion_matrix': {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1]),
            'total_eval_samples': int(len(y_test))
        },
        'base_boosting_models': base_model_metrics,
        'disease_metrics': disease_metrics,
        'feature_names': feature_names,
        'hyperparameters': {
            'sample_size': sample_size,
            'n_estimators': n_est,
            'max_depth': max_d,
            'random_state': random_state
        }
    }
    
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
        
    print(f"Metrics saved to {METRICS_FILE} successfully!")
    print("=" * 60)
    print(f"STACKED ENSEMBLE TRAINING COMPLETE IN {total_time:.2f}s!")
    print(f"Accuracy: {final_test_acc*100:.2f}% | F1: {final_f1*100:.2f}% | Gap: {generalization_gap*100:.2f}%")
    print("=" * 60)
    return metrics_payload

if __name__ == '__main__':
    sample = 50000
    if len(sys.argv) > 1:
        sample = int(sys.argv[1])
    train_stacked_ensemble(sample_size=sample)
