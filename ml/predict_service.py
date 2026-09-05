import os
import sys
import json
import time
import joblib
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
MODEL_FILE = os.path.join(CACHE_DIR, "stacked_ensemble.joblib")
METRICS_FILE = os.path.join(CACHE_DIR, "evaluation_metrics.json")

_bundle_cache = None

def get_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Please run train_ensemble.py first.")
        _bundle_cache = joblib.load(MODEL_FILE)
    return _bundle_cache

def calculate_bmi_level(bmi):
    if bmi < 18.5:
        return 0 # Underweight
    elif bmi < 25.0:
        return 1 # Normal
    elif bmi < 30.0:
        return 2 # Overweight
    else:
        return 3 # Obese

def parse_patient_vector(patient):
    # Defaults and bounds
    age = float(patient.get('age', 45))
    gender = str(patient.get('gender', 'Female')).strip().capitalize()
    gender_code = 1 if gender == 'Male' else 0
    
    bmi = float(patient.get('bmi', 26.5))
    bmi_code = calculate_bmi_level(bmi)
    
    systolic_bp = float(patient.get('systolic_bp', 125))
    diastolic_bp = float(patient.get('diastolic_bp', 82))
    heart_rate = float(patient.get('heart_rate', 72))
    
    glucose = float(patient.get('glucose', 105))
    HbA1c = float(patient.get('HbA1c_level', 5.6))
    cholesterol = float(patient.get('cholesterol', 215))
    triglycerides = float(patient.get('triglycerides', 160))
    hdl = float(patient.get('hdl', 52))
    ldl = float(patient.get('ldl', 125))
    
    crp_level = float(patient.get('crp_level', 3.5))
    homocysteine_level = float(patient.get('homocysteine_level', 11.5))
    
    # Lifestyle mappings
    smoking_map = {'never': 0, 'former': 1, 'current': 2}
    smoking_val = str(patient.get('smoking', 'Never')).lower()
    smoking_code = smoking_map.get(smoking_val, 0)
    
    alcohol_intake = float(patient.get('alcohol_intake', 5.0))
    salt_intake = float(patient.get('salt_intake', 6.0))
    
    sugar_map = {'low': 0, 'medium': 1, 'high': 2}
    sugar_code = sugar_map.get(str(patient.get('sugar_consumption', 'Medium')).lower(), 1)
    
    activity_map = {'low': 0, 'moderate': 1, 'high': 2}
    activity_code = activity_map.get(str(patient.get('physical_activity', 'Moderate')).lower(), 1)
    
    sleep_hours = float(patient.get('sleep_hours', 7.0))
    family_code = 1 if str(patient.get('family_history', 'No')).lower() in ['yes', 'true', '1'] else 0
    
    stress_map = {'low': 0, 'medium': 1, 'high': 2}
    stress_code = stress_map.get(str(patient.get('stress_level', 'Medium')).lower(), 1)
    
    edu_map = {'primary': 0, 'secondary': 1, 'tertiary': 2}
    edu_code = edu_map.get(str(patient.get('education_level', 'Secondary')).lower(), 1)
    
    emp_map = {'unemployed': 0, 'employed': 1, 'retired': 2}
    emp_code = emp_map.get(str(patient.get('employment_status', 'Employed')).lower(), 1)
    
    vector = [
        age, gender_code, bmi, bmi_code,
        systolic_bp, diastolic_bp, heart_rate,
        glucose, HbA1c, cholesterol, triglycerides, hdl, ldl,
        crp_level, homocysteine_level,
        smoking_code, alcohol_intake, salt_intake, sugar_code,
        activity_code, sleep_hours, family_code, stress_code,
        edu_code, emp_code
    ]
    return np.array(vector, dtype=float).reshape(1, -1), patient

def predict_patient(patient_data):
    start_time = time.time()
    bundle = get_bundle()
    
    base_models = bundle['base_models']
    meta_rf = bundle['meta_model']
    disease_calibrators = bundle.get('disease_calibrators', {})
    
    X_sample, raw_patient = parse_patient_vector(patient_data)
    
    # 1. Base boosting predictions
    boosting_probs = {}
    meta_features = []
    
    for name, model in base_models.items():
        proba = float(model.predict_proba(X_sample)[0, 1])
        boosting_probs[name] = round(proba * 100, 2)
        meta_features.append(proba)
        
    X_meta = np.array(meta_features).reshape(1, -1)
    
    # 2. Meta Random Forest stacked prediction
    stacked_meta_proba = float(meta_rf.predict_proba(X_meta)[0, 1])
    stacked_confidence = round(stacked_meta_proba * 100, 2)
    overall_prediction = "Abnormal / Elevated Risk" if stacked_meta_proba >= 0.5 else "Normal / Stable Profile"
    
    # 3. Predict all 10 target diseases with confidence scores
    diseases_config = [
        {
            "id": "diabetes",
            "name": "Type 2 Diabetes",
            "category": "Endocrine & Metabolic",
            "icon": "fa-syringe",
            "description": "Chronic condition affecting cellular glucose uptake and blood sugar regulation.",
            "biomarkers": ["HbA1c", "Fasting Glucose"]
        },
        {
            "id": "prediabetes",
            "name": "Prediabetes",
            "category": "Endocrine & Metabolic",
            "icon": "fa-notes-medical",
            "description": "Elevated blood sugar levels hovering just below the clinical diagnostic threshold for diabetes.",
            "biomarkers": ["HbA1c 5.7-6.4%", "Fasting Glucose 100-125 mg/dL"]
        },
        {
            "id": "hypertension",
            "name": "Hypertension",
            "category": "Cardiovascular",
            "icon": "fa-heart-pulse",
            "description": "Sustained arterial blood pressure elevation placing chronic stress on blood vessels.",
            "biomarkers": ["Systolic BP >= 130 mmHg", "Diastolic BP >= 80 mmHg"]
        },
        {
            "id": "heart_disease",
            "name": "Coronary Heart Disease",
            "category": "Cardiovascular",
            "icon": "fa-heart-crack",
            "description": "Plaque accumulation or arterial stiffening restricting cardiac oxygen perfusion.",
            "biomarkers": ["Homocysteine", "CRP", "LDL/HDL Ratio", "Resting Heart Rate"]
        },
        {
            "id": "kidney_disease",
            "name": "Kidney Disease (CKD/AKI)",
            "category": "Renal Function",
            "icon": "fa-vial-circle-check",
            "description": "Progressive decrease in glomerular filtration and microvascular renal filtration capacity.",
            "biomarkers": ["Homocysteine > 14", "CRP > 6.0", "Systolic BP > 140", "HbA1c > 7.0"]
        },
        {
            "id": "liver_disease",
            "name": "Hepatic Steatosis / MASLD",
            "category": "Hepatic Health",
            "icon": "fa-shield-halved",
            "description": "Excess hepatic lipid deposition and inflammatory liver parenchyma strain.",
            "biomarkers": ["Triglycerides > 200", "BMI >= 29", "Alcohol Intake", "Sugar Intake"]
        },
        {
            "id": "anemia",
            "name": "Anemia / Hematologic Stress",
            "category": "Hematological",
            "icon": "fa-droplet",
            "description": "Compromised oxygen-carrying capacity with systemic fatigue and compensatory tachycardia.",
            "biomarkers": ["Resting Tachycardia", "Elevated CRP", "Elevated Homocysteine"]
        },
        {
            "id": "thyroid_disorders",
            "name": "Thyroid Dysregulation",
            "category": "Endocrine & Metabolic",
            "icon": "fa-dna",
            "description": "Altered metabolic rate marked by heart rate anomalies, lipid shifts, and chronic fatigue.",
            "biomarkers": ["Heart Rate Outliers", "Dyslipidemia", "High Stress Index"]
        },
        {
            "id": "obesity",
            "name": "Obesity (Class I-III)",
            "category": "Anthropometric",
            "icon": "fa-weight-scale",
            "description": "Excess adiposity index increasing systemic mechanical and metabolic burden.",
            "biomarkers": ["BMI >= 30.0 kg/m²"]
        },
        {
            "id": "metabolic_syndrome",
            "name": "Metabolic Syndrome Complex",
            "category": "Syndromic Complex",
            "icon": "fa-diagram-project",
            "description": "Concurrent clustering of central obesity, hypertension, hypertriglyceridemia, and hyperglycemia.",
            "biomarkers": ["ATP III Criteria: 3+ of Central Obesity, BP, Glucose, Triglycerides, HDL"]
        }
    ]
    
    disease_results = []
    
    for cfg in diseases_config:
        d_id = cfg["id"]
        # Determine calibrated probability
        if d_id in disease_calibrators:
            calibrator = disease_calibrators[d_id]
            prob = float(calibrator.predict_proba(X_sample)[0, 1])
        else:
            # Fallback clinical rule engine
            prob = stacked_meta_proba
            
        # Clinical adjustment based on direct diagnostic rules
        age = float(raw_patient.get('age', 45))
        bmi = float(raw_patient.get('bmi', 25))
        glu = float(raw_patient.get('glucose', 100))
        hba1c = float(raw_patient.get('HbA1c_level', 5.5))
        sbp = float(raw_patient.get('systolic_bp', 120))
        dbp = float(raw_patient.get('diastolic_bp', 80))
        tg = float(raw_patient.get('triglycerides', 150))
        hdl = float(raw_patient.get('hdl', 50))
        crp = float(raw_patient.get('crp_level', 3.0))
        hcys = float(raw_patient.get('homocysteine_level', 11.0))
        
        # Clinical boundary calibrations
        if d_id == 'diabetes':
            if hba1c >= 6.5 or glu >= 126:
                prob = max(prob, 0.88)
            elif hba1c < 5.7 and glu < 100:
                prob = min(prob, 0.15)
        elif d_id == 'prediabetes':
            if (5.7 <= hba1c < 6.5) or (100 <= glu <= 125):
                prob = max(prob, 0.82)
            elif hba1c >= 6.5 or glu >= 126:
                prob = min(prob, 0.25)
            elif hba1c < 5.7 and glu < 100:
                prob = min(prob, 0.10)
        elif d_id == 'hypertension':
            if sbp >= 140 or dbp >= 90:
                prob = max(prob, 0.90)
            elif sbp >= 130 or dbp >= 80:
                prob = max(prob, 0.72)
            elif sbp < 120 and dbp < 80:
                prob = min(prob, 0.10)
        elif d_id == 'obesity':
            if bmi >= 30.0:
                prob = max(prob, 0.96)
            elif bmi >= 25.0:
                prob = min(max(prob, 0.45), 0.65)
            else:
                prob = min(prob, 0.05)
        elif d_id == 'heart_disease':
            if hcys > 15.0 and crp > 6.0 and sbp >= 135:
                prob = max(prob, 0.84)
            elif sbp < 120 and hcys < 10.0 and crp < 2.0:
                prob = min(prob, 0.12)
        elif d_id == 'metabolic_syndrome':
            crit = (1 if bmi >= 30.0 else 0) + (1 if tg >= 150.0 else 0) + (1 if hdl < 50.0 else 0) + (1 if sbp >= 130 or dbp >= 85 else 0) + (1 if glu >= 100.0 else 0)
            if crit >= 3:
                prob = max(prob, 0.75 + 0.07 * (crit - 3))
            else:
                prob = min(prob, 0.10 * crit)
                
        confidence = round(prob * 100, 1)
        
        # Risk level classification
        if confidence >= 80:
            risk_tier = "Critical"
            badge_class = "badge-danger"
            severity = 4
        elif confidence >= 60:
            risk_tier = "High Risk"
            badge_class = "badge-warning"
            severity = 3
        elif confidence >= 35:
            risk_tier = "Moderate Risk"
            badge_class = "badge-moderate"
            severity = 2
        elif confidence >= 15:
            risk_tier = "Low Risk"
            badge_class = "badge-low"
            severity = 1
        else:
            risk_tier = "Optimal / Clear"
            badge_class = "badge-success"
            severity = 0
            
        disease_results.append({
            **cfg,
            "confidence": confidence,
            "probability": round(prob, 4),
            "risk_tier": risk_tier,
            "badge_class": badge_class,
            "severity": severity,
            "is_positive": confidence >= 50.0
        })
        
    # Sort diseases by severity and confidence
    disease_results.sort(key=lambda x: (x["severity"], x["confidence"]), reverse=True)
    
    # Calibrate overall stacked confidence with multi-disease outcomes
    max_disease_conf = max(d["confidence"] for d in disease_results) if disease_results else 0
    if max_disease_conf < 25.0:
        overall_prediction = "Normal / Optimal Health Profile"
        stacked_confidence = round(min(stacked_meta_proba * 100 * 0.2, 14.0), 1)
    elif max_disease_conf >= 50.0:
        overall_prediction = "Abnormal / Elevated Disease Risk"
        stacked_confidence = round(max(stacked_meta_proba * 100, max_disease_conf), 1)
    else:
        overall_prediction = "Borderline / Moderate Health Vigilance"
        stacked_confidence = round(max_disease_conf, 1)
    
    # Clinical Lifestyle and Action Recommendations
    recommendations = []
    if float(raw_patient.get('glucose', 100)) >= 110 or float(raw_patient.get('HbA1c_level', 5.5)) >= 5.8:
        recommendations.append("Adopt low glycemic-index complex carbohydrates; consult an endocrinologist for OGTT and dietary restructuring.")
    if float(raw_patient.get('systolic_bp', 120)) >= 130 or float(raw_patient.get('diastolic_bp', 80)) >= 85:
        recommendations.append("Initiate the DASH diet (sodium < 2,000 mg/day) and schedule automated 24-hour ambulatory blood pressure monitoring.")
    if float(raw_patient.get('bmi', 25)) >= 30.0:
        recommendations.append("Structured metabolic intervention aiming for 7-10% body weight reduction via calorie deficit and resistance training.")
    if float(raw_patient.get('crp_level', 2.0)) > 5.0 or float(raw_patient.get('homocysteine_level', 10)) > 14:
        recommendations.append("Systemic inflammatory & cardiovascular biomarkers elevated; consider cardiology consultation, B-vitamin evaluation, and lipid optimization.")
    if str(raw_patient.get('smoking', 'Never')).lower() in ['current', 'former']:
        recommendations.append("Prioritize smoking cessation protocols to eliminate ongoing vascular endothelium injury and reduce thrombotic risk.")
    if str(raw_patient.get('physical_activity', 'Moderate')).lower() == 'low':
        recommendations.append("Incorporate at least 150 minutes of moderate-intensity aerobic physical activity per week.")
        
    if not recommendations:
        recommendations.append("All primary biomarkers are within standard clinical bounds. Maintain regular physical activity, balanced Mediterranean nutrition, and annual health screenings.")
        
    elapsed = time.time() - start_time
    
    return {
        "status": "success",
        "overall_prediction": overall_prediction,
        "stacked_confidence": stacked_confidence,
        "stacked_probability": round(stacked_meta_proba, 4),
        "execution_time_ms": round(elapsed * 1000, 1),
        "boosting_models": {
            "xgboost": boosting_probs.get("xgboost", 0),
            "lightgbm": boosting_probs.get("lightgbm", 0),
            "catboost": boosting_probs.get("catboost", 0),
            "gradient_boosting": boosting_probs.get("gradient_boosting", 0),
            "adaboost": boosting_probs.get("adaboost", 0)
        },
        "meta_learner": {
            "model_type": "RandomForestClassifier",
            "weighting_strategy": "Nonlinear Stacking on Probability Vectors",
            "stacked_score": stacked_confidence
        },
        "diseases": disease_results,
        "recommendations": recommendations,
        "patient_summary": {
            "age": raw_patient.get('age'),
            "gender": raw_patient.get('gender'),
            "bmi": raw_patient.get('bmi'),
            "bp": f"{raw_patient.get('systolic_bp', 120)}/{raw_patient.get('diastolic_bp', 80)} mmHg",
            "glucose": f"{raw_patient.get('glucose', 100)} mg/dL",
            "hba1c": f"{raw_patient.get('HbA1c_level', 5.5)}%"
        }
    }

if __name__ == '__main__':
    # Test CLI invocation
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            sample_in = json.load(f)
    else:
        # Default high-risk test profile
        sample_in = {
            "age": 58,
            "gender": "Male",
            "bmi": 32.4,
            "systolic_bp": 146,
            "diastolic_bp": 92,
            "heart_rate": 82,
            "glucose": 168,
            "HbA1c_level": 7.8,
            "cholesterol": 245,
            "triglycerides": 230,
            "hdl": 38,
            "ldl": 158,
            "crp_level": 8.2,
            "homocysteine_level": 15.4,
            "smoking": "Current",
            "alcohol_intake": 16.0,
            "salt_intake": 9.2,
            "sugar_consumption": "High",
            "physical_activity": "Low",
            "sleep_hours": 5.5,
            "family_history": "Yes",
            "stress_level": "High",
            "education_level": "Secondary",
            "employment_status": "Employed"
        }
    res = predict_patient(sample_in)
    print(json.dumps(res, indent=2))
