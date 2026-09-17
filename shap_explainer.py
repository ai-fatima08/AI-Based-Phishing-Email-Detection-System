import joblib
import os
import numpy as np
from text_preprocessor import TextPreprocessor
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

class SHAPExplainer:
    def __init__(
        self, 
        model=None, 
        vectorizer=None
        ):
        self.model        = model
        self.vectorizer   = vectorizer
        self.preprocessor = TextPreprocessor()
        self.explainer    = None
        if SHAP_AVAILABLE and model is not None:
            self._init_explainer()

    def _init_explainer(self):
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("\n--------------- SHAP TreeExplainer initialized --------------- ")
        except Exception as e:
            print(f"SHAP initialization failed: {e}")
            self.explainer = None

    def update_model(
        self, 
        model, 
        vectorizer
        ):
        self.model      = model
        self.vectorizer = vectorizer
        if SHAP_AVAILABLE and model is not None:
            self._init_explainer()

    def explain(
        self,
        email_text: str, 
        top_n: int = 30) -> dict:
        if not SHAP_AVAILABLE or self.explainer is None or self.vectorizer is None:
            return self._fallback_explanation(email_text)
        try:
            processed      = self.preprocessor.preprocess(email_text)
            features       = self.vectorizer.transform([processed])
            # Sparse matrix ko dense array mein convert karo
            features_dense = np.array(features.toarray(), dtype=np.float64)
            # calculating contribution of each feature
            shap_vals      = self.explainer.shap_values(features_dense)

            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                phishing_shap = np.array(shap_vals[1], dtype=np.float64).flatten()
            else:
                phishing_shap = np.array(shap_vals, dtype=np.float64).flatten()

            n_features    = phishing_shap.shape[0]
            feature_names = [f'feat_{i}' for i in range(n_features)]

            top_features = self._get_top_features(phishing_shap, feature_names, top_n)
            metadata     = self.preprocessor.extract_metadata(email_text)
            summary      = self._make_summary(top_features, metadata)

            exp_val = self.explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                exp_val = float(exp_val[1])
            else:
                exp_val = float(exp_val)
            return {
                'using_shap':     True,
                'top_features':   top_features,
                'metadata':       metadata,
                'summary':        summary,
                'expected_value': exp_val,
            }
        except Exception as e:
            print(f"SHAP explain error: {e}")
            return self._fallback_explanation(email_text)

    def _get_top_features(
        self, 
        shap_array, 
        feature_names,
        top_n: int) -> list:
        shap_array = np.array(shap_array, dtype=np.float64).flatten()
        abs_vals   = np.abs(shap_array)
        top_idx    = np.argsort(abs_vals)[-top_n:][::-1]
        top_idx    = [i for i in top_idx if i < len(feature_names)] 
        results = []
        for rank, idx in enumerate(top_idx, 1):
            val = float(shap_array[idx])
            if abs(val) < 1e-6:
                continue
            results.append({
                'rank':       rank,
                'feature':    str(feature_names[idx]),
                'shap_value': round(val, 6),
                'abs_value':  round(abs(val), 6),
                'impact':     'phishing' if val > 0 else 'safe',
                'strength':   self._strength_label(abs(val)),
            })
        return results

    def _strength_label(
        self,
        abs_val: float) -> str:
        if abs_val > 0.05:  return 'very strong'
        if abs_val > 0.02:  return 'strong'
        if abs_val > 0.01:  return 'moderate'
        if abs_val > 0.005: return 'weak'
        return 'minimal'

    def _make_summary(
        self,
        top_features: list, 
        metadata: dict) -> str:
        if not top_features:
            return "No significant features found...."
        phishing_f = [f for f in top_features if f['impact'] == 'phishing']
        safe_f     = [f for f in top_features if f['impact'] == 'safe']
        parts      = []
        if phishing_f:
            parts.append(f"Phishing indicators: {', '.join(f['feature'] for f in phishing_f[:7])}")
        if safe_f:
            parts.append(f"Safe indicators: {', '.join(f['feature'] for f in safe_f[:7])}")
        if metadata.get('url_count', 0) > 0:
            parts.append(f"{metadata['url_count']} URL(s) found")
        if metadata.get('uppercase_ratio', 0) > 0.3:
            parts.append("High use of UPPERCASE text")
        if metadata.get('exclamation_count', 0) > 2:
            parts.append(f"{metadata['exclamation_count']} exclamation marks")
        return "\n ".join(parts) + "\n" if parts else "Analysis complete."

    def _fallback_explanation(
        self,
        email_text: str) -> dict:
        metadata = self.preprocessor.extract_metadata(email_text)
        keywords = metadata.get('found_phishing_keywords', [])
        fallback_features = []
        for i, kw in enumerate(keywords[:5], 1):
            fallback_features.append({
                'rank': i, 'feature': kw,
                'shap_value': 0.05, 'abs_value': 0.05,
                'impact': 'phishing', 'strength': 'weak (estimated)',
            })
        return {
            'using_shap':     False,
            'top_features':   fallback_features,
            'metadata':       metadata,
            'summary':        f"Keywords found: {', '.join(keywords)}" if keywords
                              else "No phishing keywords detected.",
            'expected_value': 0.5,
        }

if __name__ == "__main__":
    MODEL_PATH = "models/rf_model.joblib"
    VEC_PATH   = "models/tfidf.joblib"
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        print("Model files not found....")
    else:
        rf    = joblib.load(MODEL_PATH)
        tfidf = joblib.load(VEC_PATH)
        exp   = SHAPExplainer(model=rf, vectorizer=tfidf)
        for email in [
            "URGENT: Your PayPal account suspended! Verify NOW at http://fake.com",
            "Hi, the team meeting is Monday at 2 PM."
        ]:
            print(f"\n {email[:60]}...")
            r = exp.explain(email, top_n=7)
            print(f"   Using SHAP : {r['using_shap']}")
            print(f"   Summary    : {r['summary']}")