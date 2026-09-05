/**
 * MegaDisease AI — Front-End Application Logic
 * Stacked Boosting (XGBoost, LightGBM, CatBoost, GradientBoosting, AdaBoost)
 * + Random Forest Meta-Learner
 */

document.addEventListener('DOMContentLoaded', () => {
  // State Management
  let currentStep = 1;
  const totalSteps = 4;
  let cachedMetrics = null;
  let presetsData = {};

  // DOM Elements
  const form = document.getElementById('patient-form');
  const stepProgressFill = document.getElementById('step-progress-fill');
  const stepItems = document.querySelectorAll('.step-item');
  const slidePanes = document.querySelectorAll('.slide-pane');
  const toast = document.getElementById('toast');

  // Diagnostics Modal Elements
  const diagModal = document.getElementById('diagnostics-modal');
  const btnOpenDiag = document.getElementById('btn-open-diagnostics');
  const btnCloseDiag = document.getElementById('btn-close-diagnostics');
  const btnViewMetricsFromResults = document.getElementById('btn-view-metrics-from-results');

  // BMI Modal Elements
  const bmiModal = document.getElementById('bmi-modal');
  const btnCalcBmi = document.getElementById('btn-calc-bmi');
  const btnCloseBmi = document.getElementById('btn-close-bmi');
  const btnCancelBmi = document.getElementById('btn-cancel-bmi');
  const btnApplyBmi = document.getElementById('btn-apply-bmi');
  const bmiHeightInput = document.getElementById('bmi-height');
  const bmiWeightInput = document.getElementById('bmi-weight');
  const bmiCalcVal = document.getElementById('bmi-calc-val');
  const bmiField = document.getElementById('bmi');
  const bmiStatusBadge = document.getElementById('bmi-status');

  // Lifestyle Controls
  const toggleSmoking = document.getElementById('toggle-smoking');
  const smokingDetail = document.getElementById('smoking-detail');
  const smokingType = document.getElementById('smoking_type');
  const toggleAlcohol = document.getElementById('toggle-alcohol');
  const alcoholSlider = document.getElementById('alcohol_intake');
  const alcoholDisplay = document.getElementById('alcohol_display');
  const toggleFamily = document.getElementById('toggle-family');
  const familyHistoryInput = document.getElementById('family_history');

  // =========================================================================
  // 1. Step Navigation & Progress Indicators with Checkmarks
  // =========================================================================

  function goToStep(step) {
    if (step < 1 || step > totalSteps) return;

    // Validate fields if moving forward from Step 1, 2, or 3
    if (step > currentStep && currentStep < 4) {
      const currentPane = document.getElementById(`slide-${currentStep}`);
      const inputs = currentPane.querySelectorAll('input[required], select[required]');
      let valid = true;
      inputs.forEach(input => {
        if (!input.checkValidity()) {
          input.reportValidity();
          valid = false;
        }
      });
      if (!valid) return;
    }

    currentStep = step;
    updateStepUI();
  }

  function updateStepUI() {
    // Update Slide Panes
    slidePanes.forEach(pane => {
      const slideNum = parseInt(pane.getAttribute('data-slide'));
      if (slideNum === currentStep) {
        pane.classList.add('active');
      } else {
        pane.classList.remove('active');
      }
    });

    // Update Step Indicators and Checkmarks
    stepItems.forEach(item => {
      const stepNum = parseInt(item.getAttribute('data-step'));
      if (stepNum === currentStep) {
        item.classList.add('active');
        item.classList.remove('completed');
      } else if (stepNum < currentStep) {
        item.classList.remove('active');
        item.classList.add('completed'); // Displays animated checkmark
      } else {
        item.classList.remove('active');
        item.classList.remove('completed');
      }
    });

    // Update Progress Fill Bar
    const progressPct = ((currentStep - 1) / (totalSteps - 1)) * 100;
    stepProgressFill.style.width = `${progressPct}%`;

    // Scroll to top of slide
    window.scrollTo({ top: 120, behavior: 'smooth' });
  }

  // Next / Previous Buttons
  document.querySelectorAll('.btn-next').forEach(btn => {
    btn.addEventListener('click', () => {
      const nextStep = parseInt(btn.getAttribute('data-next'));
      goToStep(nextStep);
    });
  });

  document.querySelectorAll('.btn-prev').forEach(btn => {
    btn.addEventListener('click', () => {
      const prevStep = parseInt(btn.getAttribute('data-prev'));
      goToStep(prevStep);
    });
  });

  // Step item click navigation
  stepItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetStep = parseInt(item.getAttribute('data-step'));
      // Only allow clicking to current or completed steps
      if (targetStep <= currentStep || item.classList.contains('completed')) {
        goToStep(targetStep);
      }
    });
  });

  // Reset / New Patient
  document.getElementById('btn-restart').addEventListener('click', () => {
    form.reset();
    applyPreset('healthy');
    goToStep(1);
    showToast('Reset form to healthy baseline template.');
  });

  // =========================================================================
  // 2. Lifestyle Factor Toggles
  // =========================================================================

  toggleSmoking.addEventListener('change', () => {
    if (toggleSmoking.checked) {
      smokingDetail.classList.remove('hidden');
      smokingType.value = 'Current';
    } else {
      smokingDetail.classList.add('hidden');
      smokingType.value = 'Never';
    }
  });

  toggleAlcohol.addEventListener('change', () => {
    if (toggleAlcohol.checked) {
      alcoholSlider.disabled = false;
      if (alcoholSlider.value == 0) alcoholSlider.value = 6;
      alcoholDisplay.textContent = `${alcoholSlider.value} drinks/wk`;
    } else {
      alcoholSlider.disabled = true;
      alcoholSlider.value = 0;
      alcoholDisplay.textContent = `0 drinks/wk`;
    }
  });

  alcoholSlider.addEventListener('input', () => {
    alcoholDisplay.textContent = `${alcoholSlider.value} drinks/wk`;
  });

  toggleFamily.addEventListener('change', () => {
    familyHistoryInput.value = toggleFamily.checked ? 'Yes' : 'No';
  });

  // =========================================================================
  // 3. BMI Calculator Helper
  // =========================================================================

  function calculateBmiVal(heightCm, weightKg) {
    if (!heightCm || !weightKg || heightCm <= 0) return 24.0;
    const heightM = heightCm / 100;
    return (weightKg / (heightM * heightM)).toFixed(1);
  }

  function updateBmiBadge(bmiVal) {
    const val = parseFloat(bmiVal);
    if (val < 18.5) {
      bmiStatusBadge.textContent = 'Underweight (<18.5)';
      bmiStatusBadge.className = 'bmi-badge badge-low';
    } else if (val < 25.0) {
      bmiStatusBadge.textContent = 'Normal Weight (18.5–24.9)';
      bmiStatusBadge.className = 'bmi-badge badge-success';
    } else if (val < 30.0) {
      bmiStatusBadge.textContent = 'Overweight (25.0–29.9)';
      bmiStatusBadge.className = 'bmi-badge badge-warning';
    } else {
      bmiStatusBadge.textContent = 'Obese (≥30.0)';
      bmiStatusBadge.className = 'bmi-badge badge-danger';
    }
  }

  bmiField.addEventListener('input', () => {
    updateBmiBadge(bmiField.value);
  });

  btnCalcBmi.addEventListener('click', () => {
    bmiModal.classList.remove('hidden');
    updateBmiModalPreview();
  });

  [bmiHeightInput, bmiWeightInput].forEach(inp => {
    inp.addEventListener('input', updateBmiModalPreview);
  });

  function updateBmiModalPreview() {
    const h = parseFloat(bmiHeightInput.value);
    const w = parseFloat(bmiWeightInput.value);
    const res = calculateBmiVal(h, w);
    bmiCalcVal.textContent = `${res} kg/m²`;
  }

  btnApplyBmi.addEventListener('click', () => {
    const h = parseFloat(bmiHeightInput.value);
    const w = parseFloat(bmiWeightInput.value);
    const res = calculateBmiVal(h, w);
    bmiField.value = res;
    updateBmiBadge(res);
    bmiModal.classList.add('hidden');
    showToast(`Applied calculated BMI: ${res} kg/m²`);
  });

  [btnCloseBmi, btnCancelBmi].forEach(btn => {
    btn.addEventListener('click', () => bmiModal.classList.add('hidden'));
  });

  // =========================================================================
  // 4. Presets Engine
  // =========================================================================

  const FALLBACK_PRESETS = {
    healthy: {
      id: "healthy",
      name: "Healthy Adult (Reference)",
      data: {
        age: 28, gender: "Female", bmi: 21.8, systolic_bp: 114, diastolic_bp: 74, heart_rate: 68,
        glucose: 88, HbA1c_level: 5.1, cholesterol: 178, triglycerides: 110, hdl: 62, ldl: 94,
        crp_level: 1.2, homocysteine_level: 8.4, smoking: "Never", alcohol_intake: 2.0, salt_intake: 4.5,
        sugar_consumption: "Low", physical_activity: "High", sleep_hours: 8.0, family_history: "No",
        stress_level: "Low", education_level: "Tertiary", employment_status: "Employed"
      }
    },
    diabetic_risk: {
      id: "diabetic_risk",
      name: "Diabetic & Glycemic Risk",
      data: {
        age: 52, gender: "Male", bmi: 29.4, systolic_bp: 138, diastolic_bp: 86, heart_rate: 76,
        glucose: 172, HbA1c_level: 7.4, cholesterol: 232, triglycerides: 215, hdl: 42, ldl: 147,
        crp_level: 4.8, homocysteine_level: 13.2, smoking: "Former", alcohol_intake: 6.0, salt_intake: 7.5,
        sugar_consumption: "High", physical_activity: "Low", sleep_hours: 6.0, family_history: "Yes",
        stress_level: "High", education_level: "Secondary", employment_status: "Employed"
      }
    },
    heart_risk: {
      id: "heart_risk",
      name: "Hypertension & Cardiac Risk",
      data: {
        age: 64, gender: "Male", bmi: 28.1, systolic_bp: 156, diastolic_bp: 94, heart_rate: 84,
        glucose: 118, HbA1c_level: 6.0, cholesterol: 268, triglycerides: 220, hdl: 36, ldl: 188,
        crp_level: 7.6, homocysteine_level: 16.8, smoking: "Current", alcohol_intake: 14.0, salt_intake: 9.0,
        sugar_consumption: "Medium", physical_activity: "Low", sleep_hours: 5.5, family_history: "Yes",
        stress_level: "High", education_level: "Secondary", employment_status: "Retired"
      }
    },
    metabolic_complex: {
      id: "metabolic_complex",
      name: "Metabolic Syndrome Complex",
      data: {
        age: 48, gender: "Female", bmi: 34.6, systolic_bp: 142, diastolic_bp: 90, heart_rate: 79,
        glucose: 128, HbA1c_level: 6.3, cholesterol: 240, triglycerides: 245, hdl: 38, ldl: 153,
        crp_level: 6.1, homocysteine_level: 13.9, smoking: "Never", alcohol_intake: 4.0, salt_intake: 8.0,
        sugar_consumption: "High", physical_activity: "Low", sleep_hours: 6.5, family_history: "Yes",
        stress_level: "Medium", education_level: "Tertiary", employment_status: "Employed"
      }
    },
    renal_hepatic: {
      id: "renal_hepatic",
      name: "Renal & Hepatic Stress Alert",
      data: {
        age: 60, gender: "Male", bmi: 30.5, systolic_bp: 148, diastolic_bp: 92, heart_rate: 82,
        glucose: 142, HbA1c_level: 6.8, cholesterol: 250, triglycerides: 280, hdl: 39, ldl: 155,
        crp_level: 8.9, homocysteine_level: 17.5, smoking: "Current", alcohol_intake: 18.0, salt_intake: 9.5,
        sugar_consumption: "High", physical_activity: "Low", sleep_hours: 5.0, family_history: "Yes",
        stress_level: "High", education_level: "Primary", employment_status: "Retired"
      }
    }
  };

  async function loadPresets() {
    presetsData = FALLBACK_PRESETS;
    try {
      const res = await fetch('/api/presets');
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success') {
          presetsData = data.presets;
        }
      }
    } catch (e) {
      console.log('Using static preset templates for Netlify Drop mode');
    }
  }

  function applyPreset(presetKey) {
    const preset = presetsData[presetKey];
    if (!preset) return;

    const d = preset.data;
    // Slide 1
    if (d.age !== undefined) document.getElementById('age').value = d.age;
    if (d.gender !== undefined) document.getElementById('gender').value = d.gender;
    if (d.bmi !== undefined) {
      bmiField.value = d.bmi;
      updateBmiBadge(d.bmi);
    }
    if (d.education_level) document.getElementById('education_level').value = d.education_level;
    if (d.employment_status) document.getElementById('employment_status').value = d.employment_status;
    if (d.heart_rate) document.getElementById('heart_rate').value = d.heart_rate;

    // Slide 2
    if (d.systolic_bp) document.getElementById('systolic_bp').value = d.systolic_bp;
    if (d.diastolic_bp) document.getElementById('diastolic_bp').value = d.diastolic_bp;
    if (d.glucose) document.getElementById('glucose').value = d.glucose;
    if (d.HbA1c_level) document.getElementById('HbA1c_level').value = d.HbA1c_level;
    if (d.cholesterol) document.getElementById('cholesterol').value = d.cholesterol;
    if (d.triglycerides) document.getElementById('triglycerides').value = d.triglycerides;
    if (d.hdl) document.getElementById('hdl').value = d.hdl;
    if (d.ldl) document.getElementById('ldl').value = d.ldl;
    if (d.crp_level) document.getElementById('crp_level').value = d.crp_level;
    if (d.homocysteine_level) document.getElementById('homocysteine_level').value = d.homocysteine_level;

    // Slide 3
    if (d.smoking) {
      const isSmoker = d.smoking !== 'Never';
      toggleSmoking.checked = isSmoker;
      if (isSmoker) {
        smokingDetail.classList.remove('hidden');
        smokingType.value = d.smoking;
      } else {
        smokingDetail.classList.add('hidden');
        smokingType.value = 'Never';
      }
    }

    if (d.alcohol_intake !== undefined) {
      const hasAlcohol = d.alcohol_intake > 0;
      toggleAlcohol.checked = hasAlcohol;
      alcoholSlider.disabled = !hasAlcohol;
      alcoholSlider.value = d.alcohol_intake;
      alcoholDisplay.textContent = `${d.alcohol_intake} drinks/wk`;
    }

    if (d.physical_activity) {
      document.getElementById('physical_activity').value = d.physical_activity;
      document.getElementById('toggle-activity').checked = d.physical_activity !== 'Low';
    }

    if (d.family_history) {
      const hasFam = d.family_history === 'Yes';
      toggleFamily.checked = hasFam;
      familyHistoryInput.value = d.family_history;
    }

    if (d.sleep_hours) document.getElementById('sleep_hours').value = d.sleep_hours;
    if (d.stress_level) document.getElementById('stress_level').value = d.stress_level;
    if (d.sugar_consumption) document.getElementById('sugar_consumption').value = d.sugar_consumption;
    if (d.salt_intake) document.getElementById('salt_intake').value = d.salt_intake;

    // Highlight preset button
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-preset') === presetKey);
    });

    showToast(`Loaded preset: ${preset.name}`);
  }

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const pKey = btn.getAttribute('data-preset');
      applyPreset(pKey);
    });
  });

  // =========================================================================
  // 5. Prediction Form Submission & Results Rendering
  // =========================================================================

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Advance to Results Slide immediately with loading animation
    goToStep(4);
    const container = document.getElementById('disease-cards-container');
    container.innerHTML = `
      <div class="loading-state" style="grid-column: 1 / -1; padding: 50px 20px; text-align: center;">
        <i class="fa-solid fa-spinner fa-spin fa-3x" style="color: var(--accent-cyan); margin-bottom: 16px;"></i>
        <h3 style="color: #fff; margin-bottom: 8px;">Running 5 Boosting Models + Meta-Learner...</h3>
        <p style="color: var(--text-secondary); max-width: 500px; margin: 0 auto;">
          Evaluating XGBoost, LightGBM, CatBoost, GradientBoosting, and AdaBoost probability distributions,
          stacked through the Random Forest meta-learner with full biomarker profiling.
        </p>
      </div>
    `;

    // Construct Payload
    const payload = {
      age: parseFloat(document.getElementById('age').value),
      gender: document.getElementById('gender').value,
      bmi: parseFloat(document.getElementById('bmi').value),
      education_level: document.getElementById('education_level').value,
      employment_status: document.getElementById('employment_status').value,
      heart_rate: parseFloat(document.getElementById('heart_rate').value),

      systolic_bp: parseFloat(document.getElementById('systolic_bp').value),
      diastolic_bp: parseFloat(document.getElementById('diastolic_bp').value),
      glucose: parseFloat(document.getElementById('glucose').value),
      HbA1c_level: parseFloat(document.getElementById('HbA1c_level').value),
      cholesterol: parseFloat(document.getElementById('cholesterol').value),
      triglycerides: parseFloat(document.getElementById('triglycerides').value),
      hdl: parseFloat(document.getElementById('hdl').value),
      ldl: parseFloat(document.getElementById('ldl').value),
      crp_level: parseFloat(document.getElementById('crp_level').value),
      homocysteine_level: parseFloat(document.getElementById('homocysteine_level').value),

      smoking: toggleSmoking.checked ? smokingType.value : 'Never',
      alcohol_intake: toggleAlcohol.checked ? parseFloat(alcoholSlider.value) : 0,
      physical_activity: document.getElementById('physical_activity').value,
      family_history: toggleFamily.checked ? 'Yes' : 'No',
      sleep_hours: parseFloat(document.getElementById('sleep_hours').value),
      stress_level: document.getElementById('stress_level').value,
      sugar_consumption: document.getElementById('sugar_consumption').value,
      salt_intake: parseFloat(document.getElementById('salt_intake').value)
    };

    let result = null;

    // Try backend API first (when running with Node.js)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (response.ok) {
        result = await response.json();
      }
    } catch (err) {
      console.log('Backend not detected or static hosting (Netlify Drop) active. Using Client ML Engine.');
    }

    // Seamless fallback to 100% Client-Side Machine Learning Engine (Netlify Drop)
    if (!result || result.status !== 'success') {
      if (window.ClientMLEngine) {
        result = window.ClientMLEngine.predict(payload);
        showToast('Running on Client-Side Stacked ML Engine (Netlify Drop Mode)');
      }
    }

    if (result && result.status === 'success') {
      renderPredictionResults(result);
    } else {
      container.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--accent-crimson);">
          <i class="fa-solid fa-triangle-exclamation fa-2x"></i>
          <p style="margin-top: 10px;">Prediction Error: Unable to process prediction</p>
        </div>
      `;
    }
  });

  function renderPredictionResults(data) {
    // 1. Overall Banner & Stacked Score
    const stackedConfidence = data.stacked_confidence || 0;
    document.getElementById('stacked-score').textContent = `${stackedConfidence}%`;
    const isAtRisk = stackedConfidence >= 50;

    const bannerText = document.getElementById('overall-status-text');
    const bannerSub = document.getElementById('overall-status-sub');
    const ring = document.getElementById('status-ring');

    if (stackedConfidence >= 80) {
      bannerText.textContent = "High Systemic Disease Risk Detected";
      bannerSub.textContent = `Multiple elevated cardiometabolic and organ strain biomarkers detected (Runtime: ${data.execution_time_ms} ms)`;
      ring.style.borderColor = 'var(--accent-crimson)';
      ring.style.color = 'var(--accent-crimson)';
      ring.style.background = 'rgba(239, 68, 68, 0.15)';
    } else if (stackedConfidence >= 40) {
      bannerText.textContent = "Moderate / Elevated Health Vigilance";
      bannerSub.textContent = `Borderline biomarkers detected warranting proactive clinical lifestyle intervention (Runtime: ${data.execution_time_ms} ms)`;
      ring.style.borderColor = 'var(--accent-amber)';
      ring.style.color = 'var(--accent-amber)';
      ring.style.background = 'rgba(245, 158, 11, 0.15)';
    } else {
      bannerText.textContent = "Normal / Robust Health Profile";
      bannerSub.textContent = `Patient biomarkers demonstrate optimal homeostatic stability across the ensemble (Runtime: ${data.execution_time_ms} ms)`;
      ring.style.borderColor = 'var(--accent-emerald)';
      ring.style.color = 'var(--accent-emerald)';
      ring.style.background = 'rgba(16, 185, 129, 0.15)';
    }

    // 2. Base Boosting Model Probabilities Matrix
    const b = data.boosting_models || {};
    updateBar('fill-xgb', 'val-xgb', b.xgboost || 0);
    updateBar('fill-lgb', 'val-lgb', b.lightgbm || 0);
    updateBar('fill-cat', 'val-cat', b.catboost || 0);
    updateBar('fill-gb', 'val-gb', b.gradient_boosting || 0);
    updateBar('fill-ada', 'val-ada', b.adaboost || 0);

    // 3. 10 Predicted Diseases Grid
    const container = document.getElementById('disease-cards-container');
    container.innerHTML = '';
    const diseases = data.diseases || [];

    const positiveCount = diseases.filter(d => d.confidence >= 50).length;
    document.getElementById('positive-count').textContent = `${positiveCount} Diseases Flagged`;

    diseases.forEach(d => {
      let barClass = 'bar-clear';
      if (d.confidence >= 80) barClass = 'bar-critical';
      else if (d.confidence >= 60) barClass = 'bar-high';
      else if (d.confidence >= 35) barClass = 'bar-moderate';
      else if (d.confidence >= 15) barClass = 'bar-low';

      const card = document.createElement('div');
      card.className = 'disease-card';
      card.innerHTML = `
        <div class="card-top">
          <div class="card-title-group">
            <div class="disease-icon ${d.badge_class}">
              <i class="fa-solid ${d.icon}"></i>
            </div>
            <div>
              <div class="disease-name">${d.name}</div>
              <div class="disease-cat">${d.category}</div>
            </div>
          </div>
          <span class="badge-tag ${d.badge_class}">${d.risk_tier}</span>
        </div>
        <p>${d.description}</p>
        <div class="confidence-meter">
          <div class="conf-meta">
            <span class="conf-label">Prediction Confidence Score:</span>
            <span class="conf-score">${d.confidence}%</span>
          </div>
          <div class="conf-track">
            <div class="conf-bar ${barClass}" style="width: ${d.confidence}%"></div>
          </div>
        </div>
        <div class="biomarkers-box">
          ${d.biomarkers.map(b => `<span class="biomarker-chip"><i class="fa-solid fa-angle-right"></i> ${b}</span>`).join('')}
        </div>
      `;
      container.appendChild(card);
    });

    // 4. Clinical Recommendations
    const recList = document.getElementById('recommendations-list');
    recList.innerHTML = '';
    (data.recommendations || []).forEach(r => {
      const li = document.createElement('li');
      li.textContent = r;
      recList.appendChild(li);
    });
  }

  function updateBar(barId, valId, score) {
    const el = document.getElementById(barId);
    const val = document.getElementById(valId);
    if (el) el.style.width = `${score}%`;
    if (val) val.textContent = `${score}%`;
  }

  // =========================================================================
  // 6. Diagnostics Modal: Evaluation Metrics, Confusion Matrix & Overfitting
  // =========================================================================

  async function fetchAndRenderMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success') {
          cachedMetrics = data.metrics;
          renderMetricsView(data.metrics);
          return;
        }
      }
    } catch (e) {
      console.log('Using pre-bundled model metrics for static Netlify Drop deployment.');
    }

    if (window.MODEL_BUNDLE && window.MODEL_BUNDLE.metrics) {
      cachedMetrics = window.MODEL_BUNDLE.metrics;
      renderMetricsView(window.MODEL_BUNDLE.metrics);
    }
  }

  function renderMetricsView(m) {
    const s = m.summary || {};
    const cm = m.confusion_matrix || {};

    // 1. Overfitting & Underfitting Diagnostic Banner
    document.getElementById('overfit-status-text').textContent = s.overfitting_status || 'Optimal Generalization';
    document.getElementById('overfit-desc-text').textContent = s.overfitting_description || '';
    document.getElementById('metric-train-acc').textContent = `${((s.train_accuracy || 0) * 100).toFixed(2)}%`;
    document.getElementById('metric-test-acc').textContent = `${((s.test_accuracy || 0) * 100).toFixed(2)}%`;
    document.getElementById('metric-gap').textContent = `${((s.generalization_gap || 0) * 100).toFixed(2)}%`;

    // 2. Primary KPI Cards
    document.getElementById('kpi-accuracy').textContent = `${((s.test_accuracy || 0) * 100).toFixed(2)}%`;
    document.getElementById('kpi-precision').textContent = `${((s.precision || 0) * 100).toFixed(2)}%`;
    document.getElementById('kpi-recall').textContent = `${((s.recall || 0) * 100).toFixed(2)}%`;
    document.getElementById('kpi-f1').textContent = `${((s.f1_score || 0) * 100).toFixed(2)}%`;

    // 3. Confusion Matrix Heatmap
    const total = cm.total_eval_samples || 1;
    const tn = cm.true_negative || 0;
    const fp = cm.false_positive || 0;
    const fn = cm.false_negative || 0;
    const tp = cm.true_positive || 0;

    document.getElementById('cm-tn').textContent = tn.toLocaleString();
    document.getElementById('cm-tn-pct').textContent = `${((tn / total) * 100).toFixed(1)}% of holdout`;

    document.getElementById('cm-fp').textContent = fp.toLocaleString();
    document.getElementById('cm-fp-pct').textContent = `${((fp / total) * 100).toFixed(1)}% of holdout`;

    document.getElementById('cm-fn').textContent = fn.toLocaleString();
    document.getElementById('cm-fn-pct').textContent = `${((fn / total) * 100).toFixed(1)}% of holdout`;

    document.getElementById('cm-tp').textContent = tp.toLocaleString();
    document.getElementById('cm-tp-pct').textContent = `${((tp / total) * 100).toFixed(1)}% of holdout`;

    // 4. Boosting Models Benchmark Table
    const tbody = document.getElementById('boosting-table-body');
    tbody.innerHTML = '';
    const b = m.base_boosting_models || {};
    const names = [
      { id: 'xgboost', label: 'XGBoost' },
      { id: 'lightgbm', label: 'LightGBM' },
      { id: 'catboost', label: 'CatBoost' },
      { id: 'gradient_boosting', label: 'GradientBoosting' },
      { id: 'adaboost', label: 'AdaBoost' }
    ];

    names.forEach(n => {
      const modelData = b[n.id] || {};
      const tr = document.createElement('tr');
      const trAcc = ((modelData.train_accuracy || 0) * 100).toFixed(1);
      const teAcc = ((modelData.test_accuracy || 0) * 100).toFixed(1);
      const f1 = ((modelData.f1_score || 0) * 100).toFixed(1);
      const gap = ((modelData.generalization_gap || 0) * 100).toFixed(2);

      tr.innerHTML = `
        <td>${n.label}</td>
        <td>${trAcc}%</td>
        <td>${teAcc}%</td>
        <td>${f1}%</td>
        <td><span style="color: ${gap < 1.0 ? 'var(--accent-emerald)' : 'var(--accent-amber)'}">${gap}%</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Diagnostics Modal Open / Close
  btnOpenDiag.addEventListener('click', () => {
    diagModal.classList.remove('hidden');
    fetchAndRenderMetrics();
  });

  btnViewMetricsFromResults.addEventListener('click', () => {
    diagModal.classList.remove('hidden');
    fetchAndRenderMetrics();
  });

  btnCloseDiag.addEventListener('click', () => {
    diagModal.classList.add('hidden');
  });

  // Close modals on backdrop click
  [diagModal, bmiModal].forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });
  });

  // =========================================================================
  // 7. Interactive Model Fine-Tuning Execution
  // =========================================================================

  const btnRunFineTune = document.getElementById('btn-run-finetune');
  const fineTuneStatus = document.getElementById('finetune-status');
  const ftSampleSelect = document.getElementById('ft-sample');

  btnRunFineTune.addEventListener('click', async () => {
    const sampleSize = parseInt(ftSampleSelect.value);
    btnRunFineTune.disabled = true;
    btnRunFineTune.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Fine-Tuning (${sampleSize} rows)...`;
    fineTuneStatus.classList.remove('hidden');
    fineTuneStatus.innerHTML = `Running cross-boosting calibration & Random Forest meta-stacking... please allow ~30 seconds.`;

    try {
      const res = await fetch('/api/finetune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_size: sampleSize })
      });
      const data = await res.json();
      if (data.status === 'success') {
        fineTuneStatus.innerHTML = `<i class="fa-solid fa-check"></i> Fine-tuning complete! Accuracy: ${(data.metrics.summary.test_accuracy*100).toFixed(2)}%`;
        renderMetricsView(data.metrics);
        showToast('Ensemble fine-tuned successfully!');
      } else {
        fineTuneStatus.innerHTML = `<span style="color: var(--accent-crimson);">Error: ${data.message}</span>`;
      }
    } catch (e) {
      fineTuneStatus.innerHTML = `<span style="color: var(--accent-crimson);">Network error: ${e.message}</span>`;
    } finally {
      btnRunFineTune.disabled = false;
      btnRunFineTune.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i> Re-Train & Fine-Tune`;
    }
  });

  // =========================================================================
  // 8. Toast Helper
  // =========================================================================

  function showToast(msg) {
    toast.innerHTML = `<i class="fa-solid fa-circle-info" style="color: var(--accent-cyan);"></i> ${msg}`;
    toast.classList.remove('hidden');
    setTimeout(() => {
      toast.classList.add('hidden');
    }, 3200);
  }

  // Initial Boot
  loadPresets().then(() => {
    applyPreset('healthy');
  });
  fetchAndRenderMetrics();
  updateStepUI();
});
