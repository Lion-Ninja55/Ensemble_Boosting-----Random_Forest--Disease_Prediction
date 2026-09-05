const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');
const { spawn, execFile } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

const METRICS_PATH = path.join(__dirname, 'ml', 'cache', 'evaluation_metrics.json');
const PREDICT_SCRIPT = path.join(__dirname, 'ml', 'predict_service.py');
const TRAIN_SCRIPT = path.join(__dirname, 'ml', 'train_ensemble.py');

// Patient Profiles Presets
const PRESETS = {
  healthy: {
    id: "healthy",
    name: "Healthy Adult (Reference)",
    badge: "Low Risk Profile",
    badgeClass: "badge-success",
    description: "Young-adult reference with normal hemodynamic, metabolic, and inflammatory biomarkers.",
    data: {
      age: 28,
      gender: "Female",
      bmi: 21.8,
      systolic_bp: 114,
      diastolic_bp: 74,
      heart_rate: 68,
      glucose: 88,
      HbA1c_level: 5.1,
      cholesterol: 178,
      triglycerides: 110,
      hdl: 62,
      ldl: 94,
      crp_level: 1.2,
      homocysteine_level: 8.4,
      smoking: "Never",
      alcohol_intake: 2.0,
      salt_intake: 4.5,
      sugar_consumption: "Low",
      physical_activity: "High",
      sleep_hours: 8.0,
      family_history: "No",
      stress_level: "Low",
      education_level: "Tertiary",
      employment_status: "Employed"
    }
  },
  diabetic_risk: {
    id: "diabetic_risk",
    name: "Diabetic & Glycemic Risk",
    badge: "Endocrine Alert",
    badgeClass: "badge-danger",
    description: "Middle-aged patient with hyperglycemia, elevated HbA1c, and overweight anthropometrics.",
    data: {
      age: 52,
      gender: "Male",
      bmi: 29.4,
      systolic_bp: 138,
      diastolic_bp: 86,
      heart_rate: 76,
      glucose: 172,
      HbA1c_level: 7.4,
      cholesterol: 232,
      triglycerides: 215,
      hdl: 42,
      ldl: 147,
      crp_level: 4.8,
      homocysteine_level: 13.2,
      smoking: "Former",
      alcohol_intake: 6.0,
      salt_intake: 7.5,
      sugar_consumption: "High",
      physical_activity: "Low",
      sleep_hours: 6.0,
      family_history: "Yes",
      stress_level: "High",
      education_level: "Secondary",
      employment_status: "Employed"
    }
  },
  heart_risk: {
    id: "heart_risk",
    name: "Hypertension & Cardiac Risk",
    badge: "Cardiovascular Alert",
    badgeClass: "badge-danger",
    description: "Senior patient with sustained systolic hypertension, hyperlipidemia, and elevated homocysteine.",
    data: {
      age: 64,
      gender: "Male",
      bmi: 28.1,
      systolic_bp: 156,
      diastolic_bp: 94,
      heart_rate: 84,
      glucose: 118,
      HbA1c_level: 6.0,
      cholesterol: 268,
      triglycerides: 220,
      hdl: 36,
      ldl: 188,
      crp_level: 7.6,
      homocysteine_level: 16.8,
      smoking: "Current",
      alcohol_intake: 14.0,
      salt_intake: 9.0,
      sugar_consumption: "Medium",
      physical_activity: "Low",
      sleep_hours: 5.5,
      family_history: "Yes",
      stress_level: "High",
      education_level: "Secondary",
      employment_status: "Retired"
    }
  },
  metabolic_complex: {
    id: "metabolic_complex",
    name: "Metabolic Syndrome Complex",
    badge: "Multi-System Risk",
    badgeClass: "badge-warning",
    description: "Patient meeting multiple ATP III criteria: visceral adiposity, high triglycerides, low HDL, elevated BP.",
    data: {
      age: 48,
      gender: "Female",
      bmi: 34.6,
      systolic_bp: 142,
      diastolic_bp: 90,
      heart_rate: 79,
      glucose: 128,
      HbA1c_level: 6.3,
      cholesterol: 240,
      triglycerides: 245,
      hdl: 38,
      ldl: 153,
      crp_level: 6.1,
      homocysteine_level: 13.9,
      smoking: "Never",
      alcohol_intake: 4.0,
      salt_intake: 8.0,
      sugar_consumption: "High",
      physical_activity: "Low",
      sleep_hours: 6.5,
      family_history: "Yes",
      stress_level: "Medium",
      education_level: "Tertiary",
      employment_status: "Employed"
    }
  },
  renal_hepatic: {
    id: "renal_hepatic",
    name: "Renal & Hepatic Stress Alert",
    badge: "Organ Strain",
    badgeClass: "badge-warning",
    description: "High systemic inflammation (CRP > 8 mg/L), microvascular strain, and elevated triglycerides.",
    data: {
      age: 60,
      gender: "Male",
      bmi: 30.5,
      systolic_bp: 148,
      diastolic_bp: 92,
      heart_rate: 82,
      glucose: 142,
      HbA1c_level: 6.8,
      cholesterol: 250,
      triglycerides: 280,
      hdl: 39,
      ldl: 155,
      crp_level: 8.9,
      homocysteine_level: 17.5,
      smoking: "Current",
      alcohol_intake: 18.0,
      salt_intake: 9.5,
      sugar_consumption: "High",
      physical_activity: "Low",
      sleep_hours: 5.0,
      family_history: "Yes",
      stress_level: "High",
      education_level: "Primary",
      employment_status: "Retired"
    }
  }
};

// API: Get Preset Profiles
app.get('/api/presets', (req, res) => {
  res.json({ status: 'success', presets: PRESETS });
});

// API: Get Ensemble Evaluation Metrics & Overfitting Diagnostics
app.get('/api/metrics', (req, res) => {
  if (!fs.existsSync(METRICS_PATH)) {
    return res.status(503).json({
      status: 'pending',
      message: 'Ensemble training in progress or not yet initialized.'
    });
  }
  try {
    const raw = fs.readFileSync(METRICS_PATH, 'utf-8');
    const data = JSON.parse(raw);
    res.json({ status: 'success', metrics: data });
  } catch (err) {
    res.status(500).json({ status: 'error', message: err.message });
  }
});

// API: Predict Patient Multi-Disease Profile
app.post('/api/predict', (req, res) => {
  const patientData = req.body;
  if (!patientData || Object.keys(patientData).length === 0) {
    return res.status(400).json({ status: 'error', message: 'Missing patient health data' });
  }

  // Create temporary file with patient JSON
  const tmpPath = path.join(__dirname, 'ml', 'cache', `req_${Date.now()}_${Math.random().toString(36).substr(2, 6)}.json`);
  fs.writeFileSync(tmpPath, JSON.stringify(patientData), 'utf-8');

  const pythonCmd = 'python';
  const args = [PREDICT_SCRIPT, tmpPath];

  execFile(pythonCmd, args, { maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
    // Clean up temporary file
    try { if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath); } catch (_) {}

    if (error) {
      console.error('Prediction error:', stderr || error.message);
      return res.status(500).json({ status: 'error', message: stderr || error.message });
    }

    try {
      const parsed = JSON.parse(stdout);
      res.json(parsed);
    } catch (parseErr) {
      console.error('Failed to parse Python prediction output:', stdout);
      res.status(500).json({ status: 'error', message: 'Invalid prediction response from ML engine', raw: stdout });
    }
  });
});

// API: Fine-Tune Model Hyperparameters
app.post('/api/finetune', (req, res) => {
  const { sample_size = 40000, n_estimators = 80, max_depth = 4 } = req.body;
  console.log(`Starting on-demand fine-tune: sample_size=${sample_size}, n_est=${n_estimators}, depth=${max_depth}`);
  
  const py = spawn('python', [TRAIN_SCRIPT, String(sample_size)]);
  let stdoutData = '';
  let stderrData = '';

  py.stdout.on('data', (d) => { stdoutData += d.toString(); });
  py.stderr.on('data', (d) => { stderrData += d.toString(); });

  py.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({
        status: 'error',
        message: `Fine-tuning failed with code ${code}`,
        stderr: stderrData
      });
    }
    
    // Read updated metrics
    if (fs.existsSync(METRICS_PATH)) {
      const data = JSON.parse(fs.readFileSync(METRICS_PATH, 'utf-8'));
      return res.json({
        status: 'success',
        message: 'Stacked Ensemble successfully fine-tuned and updated!',
        metrics: data
      });
    }
    res.json({ status: 'success', message: 'Training completed', logs: stdoutData });
  });
});

// Serve frontend SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`=======================================================`);
  console.log(`Mega Disease Prediction App running on http://localhost:${PORT}`);
  console.log(`Node.js + Stacked Boosting (XGBoost, LightGBM, CatBoost, GradientBoosting, AdaBoost -> RF Meta)`);
  console.log(`=======================================================`);
});
