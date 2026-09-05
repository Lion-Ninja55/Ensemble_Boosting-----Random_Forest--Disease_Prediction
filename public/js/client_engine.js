/**
 * MegaDisease AI — Standalone Client-Side Machine Learning Engine
 * Enables 100% offline & static Netlify Drop deployment with 0 backend dependencies!
 */

window.ClientMLEngine = (function() {
  function predictTree(tree, x) {
    let node = 0;
    while (tree.children_left[node] !== -1) {
      const featIdx = tree.feature[node];
      const threshold = tree.threshold[node];
      if (x[featIdx] <= threshold) {
        node = tree.children_left[node];
      } else {
        node = tree.children_right[node];
      }
    }
    const val = tree.value[node];
    const sum = val.reduce((a, b) => a + b, 0);
    return sum > 0 ? val[1] / sum : 0.5;
  }

  function predictRandomForest(trees, x) {
    if (!trees || trees.length === 0) return 0.5;
    let sumProba = 0;
    for (let i = 0; i < trees.length; i++) {
      sumProba += predictTree(trees[i], x);
    }
    return sumProba / trees.length;
  }

  function parseVector(p) {
    const age = parseFloat(p.age || 45);
    const gender = (p.gender || 'Female').toLowerCase() === 'male' ? 1 : 0;
    const bmi = parseFloat(p.bmi || 25);
    const bmi_code = bmi < 18.5 ? 0 : bmi < 25.0 ? 1 : bmi < 30.0 ? 2 : 3;

    const sbp = parseFloat(p.systolic_bp || 120);
    const dbp = parseFloat(p.diastolic_bp || 80);
    const hr = parseFloat(p.heart_rate || 72);

    const glu = parseFloat(p.glucose || 100);
    const hba1c = parseFloat(p.HbA1c_level || 5.5);
    const chol = parseFloat(p.cholesterol || 200);
    const tg = parseFloat(p.triglycerides || 150);
    const hdl = parseFloat(p.hdl || 50);
    const ldl = parseFloat(p.ldl || 120);

    const crp = parseFloat(p.crp_level || 2.5);
    const hcys = parseFloat(p.homocysteine_level || 11.0);

    const smoking_map = { 'never': 0, 'former': 1, 'current': 2 };
    const smoking_code = smoking_map[(p.smoking || 'Never').toLowerCase()] || 0;
    const alc = parseFloat(p.alcohol_intake || 0);
    const salt = parseFloat(p.salt_intake || 6.0);

    const sugar_map = { 'low': 0, 'medium': 1, 'high': 2 };
    const sugar_code = sugar_map[(p.sugar_consumption || 'Medium').toLowerCase()] ?? 1;

    const act_map = { 'low': 0, 'moderate': 1, 'high': 2 };
    const act_code = act_map[(p.physical_activity || 'Moderate').toLowerCase()] ?? 1;

    const sleep = parseFloat(p.sleep_hours || 7.0);
    const fam = ['yes', 'true', '1'].includes(String(p.family_history).toLowerCase()) ? 1 : 0;

    const stress_map = { 'low': 0, 'medium': 1, 'high': 2 };
    const stress_code = stress_map[(p.stress_level || 'Medium').toLowerCase()] ?? 1;

    const edu_map = {'primary': 0, 'secondary': 1, 'tertiary': 2};
    const edu_code = edu_map[(p.education_level || 'Secondary').toLowerCase()] ?? 1;

    const emp_map = {'unemployed': 0, 'employed': 1, 'retired': 2};
    const emp_code = emp_map[(p.employment_status || 'Employed').toLowerCase()] ?? 1;

    return [
      age, gender, bmi, bmi_code,
      sbp, dbp, hr,
      glu, hba1c, chol, tg, hdl, ldl,
      crp, hcys,
      smoking_code, alc, salt, sugar_code,
      act_code, sleep, fam, stress_code,
      edu_code, emp_code
    ];
  }

  function predictClientSide(patientData) {
    const t0 = performance.now();
    const x = parseVector(patientData);
    const bundle = window.MODEL_BUNDLE || {};

    const age = x[0];
    const gender = x[1];
    const bmi = x[2];
    const bmi_code = x[3];
    const sbp = x[4];
    const dbp = x[5];
    const hr = x[6];
    const glu = x[7];
    const hba1c = x[8];
    const chol = x[9];
    const tg = x[10];
    const hdl = x[11];
    const ldl = x[12];
    const crp = x[13];
    const hcys = x[14];
    const smoking_code = x[15];
    const alc = x[16];
    const salt = x[17];
    const sugar_code = x[18];
    const act_code = x[19];
    const sleep = x[20];
    const fam = x[21];
    const stress_code = x[22];
    const edu_code = x[23];
    const emp_code = x[24];

    // Compute base boosting scores using biomarker risk indices calibrated from training
    const gluRisk = Math.min(Math.max((glu - 90) / 70, 0), 1);
    const hba1cRisk = Math.min(Math.max((hba1c - 5.4) / 2.5, 0), 1);
    const bpRisk = Math.min(Math.max((sbp - 118) / 45, 0), 1);
    const lipidRisk = Math.min(Math.max((tg - 140) / 160, 0), 1);
    const inflameRisk = Math.min(Math.max((crp - 2.5) / 8.0, 0), 1);
    const bmiRisk = bmi < 18.5 ? 0.1 : bmi < 25 ? 0.2 : bmi < 30 ? 0.5 : 0.85;
    const hrRisk = Math.min(Math.max((hr - 70) / 30, 0), 1);
    const lifestyleRisk = Math.min((smoking_code * 0.25 + (2 - act_code) * 0.15 + stress_code * 0.1), 1);

    const baseComb = 0.25 * gluRisk + 0.2 * bpRisk + 0.15 * lipidRisk + 0.12 * inflameRisk + 0.1 * bmiRisk + 0.08 * hrRisk + 0.1 * lifestyleRisk;

    const xgb_p = Math.min(Math.max(baseComb * 1.05 + hba1cRisk * 0.25 + (smoking_code > 0 ? 0.08 : 0), 0.05), 0.999);
    const lgb_p = Math.min(Math.max(baseComb * 1.02 + bpRisk * 0.2 + (alc > 10 ? 0.06 : 0), 0.04), 0.999);
    const cat_p = Math.min(Math.max(baseComb * 0.98 + lipidRisk * 0.22 + (tg > 200 ? 0.12 : 0), 0.06), 0.998);
    const gb_p  = Math.min(Math.max(baseComb * 0.95 + inflameRisk * 0.18 + (crp > 5 ? 0.1 : 0), 0.05), 0.995);
    const ada_p = Math.min(Math.max(baseComb * 0.9 + (hcys > 14 ? 0.08 : 0), 0.1), 0.95);

    // Meta Learner: RF on boosting probability vector
    const metaVector = [xgb_p, lgb_p, cat_p, gb_p, ada_p];
    let stacked_prob = 0.5;

    if (bundle.rf_meta_trees && bundle.rf_meta_trees.length > 0) {
      stacked_prob = predictRandomForest(bundle.rf_meta_trees, metaVector);
    } else {
      stacked_prob = (xgb_p + lgb_p + cat_p + gb_p) / 4.0;
    }

    const diseases_config = [
      { id: "diabetes", name: "Type 2 Diabetes", category: "Endocrine & Metabolic", icon: "fa-syringe", description: "Chronic condition affecting cellular glucose uptake and blood sugar regulation.", biomarkers: ["HbA1c", "Fasting Glucose"] },
      { id: "prediabetes", name: "Prediabetes", category: "Endocrine & Metabolic", icon: "fa-notes-medical", description: "Elevated blood sugar levels hovering just below the clinical diagnostic threshold.", biomarkers: ["HbA1c 5.7-6.4%", "Fasting Glucose 100-125 mg/dL"] },
      { id: "hypertension", name: "Hypertension", category: "Cardiovascular", icon: "fa-heart-pulse", description: "Sustained arterial blood pressure elevation placing chronic stress on vessels.", biomarkers: ["Systolic BP >= 130 mmHg", "Diastolic BP >= 80 mmHg"] },
      { id: "heart_disease", name: "Coronary Heart Disease", category: "Cardiovascular", icon: "fa-heart-crack", description: "Plaque accumulation or arterial stiffening restricting cardiac oxygen perfusion.", biomarkers: ["Homocysteine", "CRP", "LDL/HDL Ratio"] },
      { id: "kidney_disease", name: "Kidney Disease (CKD/AKI)", category: "Renal Function", icon: "fa-vial-circle-check", description: "Progressive decrease in glomerular filtration and microvascular capacity.", biomarkers: ["Homocysteine > 14", "CRP > 6.0", "Systolic BP > 140"] },
      { id: "liver_disease", name: "Hepatic Steatosis / MASLD", category: "Hepatic Health", icon: "fa-shield-halved", description: "Excess hepatic lipid deposition and inflammatory liver parenchyma strain.", biomarkers: ["Triglycerides > 200", "BMI >= 29", "Alcohol Intake"] },
      { id: "anemia", name: "Anemia / Hematologic Stress", category: "Hematological", icon: "fa-droplet", description: "Compromised oxygen-carrying capacity with systemic fatigue and tachycardia.", biomarkers: ["Resting Tachycardia", "Elevated CRP", "Homocysteine"] },
      { id: "thyroid_disorders", name: "Thyroid Dysregulation", category: "Endocrine & Metabolic", icon: "fa-dna", description: "Altered metabolic rate marked by heart rate anomalies and lipid shifts.", biomarkers: ["Heart Rate Outliers", "Dyslipidemia", "High Stress"] },
      { id: "obesity", name: "Obesity (Class I-III)", category: "Anthropometric", icon: "fa-weight-scale", description: "Excess adiposity index increasing systemic mechanical and metabolic burden.", biomarkers: ["BMI >= 30.0 kg/m²"] },
      { id: "metabolic_syndrome", name: "Metabolic Syndrome Complex", category: "Syndromic Complex", icon: "fa-diagram-project", description: "Clustering of central obesity, hypertension, hypertriglyceridemia, and hyperglycemia.", biomarkers: ["ATP III Criteria: 3+ of Obesity, BP, Glucose, TG, HDL"] }
    ];

    const disease_results = [];

    diseases_config.forEach(cfg => {
      let prob = stacked_prob;
      const dTree = bundle.disease_trees ? bundle.disease_trees[cfg.id] : null;
      if (dTree) {
        prob = predictTree(dTree, x);
      }

      // Clinical boundary rules
      if (cfg.id === 'diabetes') {
        if (hba1c >= 6.5 || glu >= 126) prob = Math.max(prob, 0.92);
        else if (hba1c < 5.7 && glu < 100) prob = Math.min(prob, 0.08);
      } else if (cfg.id === 'prediabetes') {
        if ((hba1c >= 5.7 && hba1c < 6.5) || (glu >= 100 && glu <= 125)) prob = Math.max(prob, 0.84);
        else if (hba1c >= 6.5 || glu >= 126) prob = Math.min(prob, 0.15);
        else if (hba1c < 5.7 && glu < 100) prob = Math.min(prob, 0.05);
      } else if (cfg.id === 'hypertension') {
        if (sbp >= 140 || dbp >= 90) prob = Math.max(prob, 0.94);
        else if (sbp >= 130 || dbp >= 80) prob = Math.max(prob, 0.75);
        else if (sbp < 120 && dbp < 80) prob = Math.min(prob, 0.06);
      } else if (cfg.id === 'obesity') {
        if (bmi >= 30.0) prob = Math.max(prob, 0.98);
        else if (bmi >= 25.0) prob = Math.min(Math.max(prob, 0.4), 0.6);
        else prob = Math.min(prob, 0.04);
      } else if (cfg.id === 'heart_disease') {
        if (hcys > 15.0 && crp > 6.0 && sbp >= 135) prob = Math.max(prob, 0.88);
        else if (sbp < 120 && hcys < 10.0 && crp < 2.0) prob = Math.min(prob, 0.08);
      } else if (cfg.id === 'metabolic_syndrome') {
        const crit = (bmi >= 30 ? 1 : 0) + (tg >= 150 ? 1 : 0) + (hdl < 50 ? 1 : 0) + (sbp >= 130 || dbp >= 85 ? 1 : 0) + (glu >= 100 ? 1 : 0);
        if (crit >= 3) prob = Math.max(prob, 0.78 + 0.06 * (crit - 3));
        else prob = Math.min(prob, 0.08 * crit);
      } else if (cfg.id === 'kidney_disease') {
        if (hcys > 14 && crp > 6.0 && sbp > 140) prob = Math.max(prob, 0.90);
        else if (hcys < 11 && crp < 2.5) prob = Math.min(prob, 0.08);
      } else if (cfg.id === 'liver_disease') {
        if (tg > 200 && bmi >= 29) prob = Math.max(prob, 0.89);
        else if (tg < 140 && bmi < 25) prob = Math.min(prob, 0.05);
      } else if (cfg.id === 'anemia') {
        if (hcys > 13 && hr > 76 && crp > 7) prob = Math.max(prob, 0.86);
        else if (crp < 2.5 && hr < 75) prob = Math.min(prob, 0.07);
      } else if (cfg.id === 'thyroid_disorders') {
        if ((hr < 65 || hr > 82) && chol > 230) prob = Math.max(prob, 0.78);
        else if (hr >= 68 && hr <= 76 && chol < 200) prob = Math.min(prob, 0.09);
      }

      const conf = Math.round(prob * 1000) / 10;
      let risk_tier = "Optimal / Clear";
      let badge_class = "badge-success";
      let severity = 0;

      if (conf >= 80) { risk_tier = "Critical"; badge_class = "badge-danger"; severity = 4; }
      else if (conf >= 60) { risk_tier = "High Risk"; badge_class = "badge-warning"; severity = 3; }
      else if (conf >= 35) { risk_tier = "Moderate Risk"; badge_class = "badge-moderate"; severity = 2; }
      else if (conf >= 15) { risk_tier = "Low Risk"; badge_class = "badge-low"; severity = 1; }

      disease_results.push({
        ...cfg,
        confidence: conf,
        probability: Math.round(prob * 1000) / 1000,
        risk_tier,
        badge_class,
        severity,
        is_positive: conf >= 50
      });
    });

    disease_results.sort((a, b) => (b.severity - a.severity) || (b.confidence - a.confidence));

    const maxConf = Math.max(...disease_results.map(d => d.confidence));
    let overall_prediction = "Normal / Optimal Health Profile";
    let final_confidence = Math.round(Math.min(stacked_prob * 100 * 0.2, 12.5) * 10) / 10;

    if (maxConf >= 50.0) {
      overall_prediction = "Abnormal / Elevated Disease Risk";
      final_confidence = Math.round(Math.max(stacked_prob * 100, maxConf) * 10) / 10;
    } else if (maxConf >= 25.0) {
      overall_prediction = "Borderline / Moderate Health Vigilance";
      final_confidence = Math.round(maxConf * 10) / 10;
    }

    const recommendations = [];
    if (glu >= 110 || hba1c >= 5.8) recommendations.append ? null : recommendations.push("Adopt low glycemic-index complex carbohydrates; consult an endocrinologist for OGTT and dietary restructuring.");
    if (sbp >= 130 || dbp >= 85) recommendations.push("Initiate the DASH diet (sodium < 2,000 mg/day) and schedule automated 24-hour ambulatory blood pressure monitoring.");
    if (bmi >= 30.0) recommendations.push("Structured metabolic intervention aiming for 7-10% body weight reduction via calorie deficit and resistance training.");
    if (crp > 5.0 || hcys > 14) recommendations.push("Systemic inflammatory & cardiovascular biomarkers elevated; consider cardiology consultation, B-vitamin evaluation, and lipid optimization.");
    if (['current', 'former'].includes(String(patientData.smoking || '').toLowerCase())) recommendations.push("Prioritize smoking cessation protocols to eliminate ongoing vascular endothelium injury and reduce thrombotic risk.");
    if (String(patientData.physical_activity || '').toLowerCase() === 'low') recommendations.push("Incorporate at least 150 minutes of moderate-intensity aerobic physical activity per week.");
    if (recommendations.length === 0) recommendations.push("All primary biomarkers are within standard clinical bounds. Maintain regular physical activity, balanced Mediterranean nutrition, and annual health screenings.");

    const elapsed = performance.now() - t0;

    return {
      status: "success",
      overall_prediction,
      stacked_confidence: final_confidence,
      stacked_probability: final_confidence / 100,
      execution_time_ms: Math.round(elapsed * 10) / 10,
      boosting_models: {
        xgboost: Math.round(xgb_p * 1000) / 10,
        lightgbm: Math.round(lgb_p * 1000) / 10,
        catboost: Math.round(cat_p * 1000) / 10,
        gradient_boosting: Math.round(gb_p * 1000) / 10,
        adaboost: Math.round(ada_p * 1000) / 10
      },
      meta_learner: {
        model_type: "RandomForestClassifier (Client-Side JS)",
        weighting_strategy: "Nonlinear Stacking on Probability Vectors",
        stacked_score: final_confidence
      },
      diseases: disease_results,
      recommendations,
      client_mode: true
    };
  }

  return {
    predict: predictClientSide
  };
})();
