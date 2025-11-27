import os
import sys
import json
import dill
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join('models', 'lsvc_calibrated.dill')
META_PATH = os.path.join('models', 'lsvc_calibrated_metadata.json')
DATA_DIR = os.path.join('.', 'data')
OUTPUTS_DIR = os.path.join('.', 'outputs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_model_and_meta(model_path=MODEL_PATH, meta_path=META_PATH):
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return model, meta

def align_features(df, meta):
    features = meta.get('features') or []
    if features:
        for c in features:
            if c not in df.columns:
                df[c] = np.nan
        return df[features]
    return df

def main(inp_csv, out_csv, threshold=0.062):
    model, meta = load_model_and_meta()
    df = pd.read_csv(inp_csv)
    X = align_features(df.copy(), meta)
    # Further processing and model prediction would go here
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        df['stroke_proba'] = probs
        df['stroke_pred'] = (probs > float(threshold)).astype(int)
        print(f"Predicted stroke probabilities for first 5 samples:", df['stroke_proba'].head().values)
        print(f'Thresholded preds (first 5):', df['stroke_pred'].head().values)
    else: 
        preds = model.predict(X)
        df['stroke_pred'] = preds
        print(f"Preds:", df['stroke_pred'].head().values)

    out_dir = os.path.dirname(out_csv) or '.'
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")

if __name__ == '__main__':
    try:
        if not (3 <= len(sys.argv) <= 4):
            print("Usage: python .\\scripts\\validate_model.py <input.csv> <output.csv> [threshold]")
            sys.exit(1)
        inp_csv = sys.argv[1]
        out_csv = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 0.062
        main(inp_csv, out_csv, threshold)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
