import os
import json
import joblib
import numpy as np

bundle_path = os.path.join(os.path.dirname(__file__), "cache", "stacked_ensemble.joblib")
metrics_path = os.path.join(os.path.dirname(__file__), "cache", "evaluation_metrics.json")
bundle = joblib.load(bundle_path)

with open(metrics_path, 'r', encoding='utf-8') as f:
    metrics = json.load(f)

def serialize_tree(tree):
    # tree is sklearn.tree._tree.Tree
    return {
        'children_left': tree.children_left.tolist(),
        'children_right': tree.children_right.tolist(),
        'feature': tree.feature.tolist(),
        'threshold': [round(float(x), 4) for x in tree.threshold],
        'value': [tree.value[i][0].tolist() for i in range(tree.node_count)]
    }

# Serialize RF Meta-Learner (top 30 estimators for compact high-speed client-side execution)
rf = bundle['meta_model']
rf_trees = [serialize_tree(est.tree_) for est in rf.estimators_[:30]]

# Serialize 10 disease calibrators
disease_trees = {}
for d_name, calibrator in bundle.get('disease_calibrators', {}).items():
    if calibrator is not None and hasattr(calibrator, 'tree_'):
        disease_trees[d_name] = serialize_tree(calibrator.tree_)

# Extract feature importances of base models
base_model_weights = {}
for name, model in bundle['base_models'].items():
    if hasattr(model, 'feature_importances_'):
        base_model_weights[name] = [round(float(x), 5) for x in model.feature_importances_]

out_data = {
    'metrics': metrics,
    'feature_names': bundle['feature_names'],
    'base_model_weights': base_model_weights,
    'rf_meta_trees': rf_trees,
    'disease_trees': disease_trees
}

out_js_path = os.path.join(os.path.dirname(__file__), "..", "public", "js", "model_bundle.js")
with open(out_js_path, 'w', encoding='utf-8') as f:
    f.write("window.MODEL_BUNDLE = " + json.dumps(out_data) + ";\n")

print(f"Successfully exported client model bundle to {out_js_path} (Size: {os.path.getsize(out_js_path) / 1024:.1f} KB)")
