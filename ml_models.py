import joblib
import numpy as np
import re
from pathlib import Path

from text_preprocessor import TextPreprocessor
from shap_explainer import SHAPExplainer as RealShapExplainer


# PHISHING KEYWORDS 
PHISHING_KEYWORDS = {
    #urgency
    'urgent': 'Urgency language detected',
    'immediately': 'Urgency language detected',
    'asap': 'Urgency language detected',
    'immediate action': 'Immediate action required',
    'urgent attention': 'Emergency language detected',
    'time sensitive': 'Time pressure detected',
    'act now': 'Immediate action demanded',
    
    # === TIME PRESSURE ===
    'expire': 'Account expiry threat',
    'expires today': 'Same-day expiration threat',
    'within 24 hours': '24-hour time pressure',
    '24 hours': 'Time pressure (24 hours)',
    '48 hours': 'Time pressure (48 hours)',
    'limited': 'Limited time pressure',
    'deadline': 'Deadline pressure',
    'today only': 'Today-only urgency',
    'last chance': 'Last chance pressure',
    #account threats
    'suspended': 'Account suspension threat',
    'locked': 'Account locked threat',
    'deleted': 'Account deletion threat',
    'permanently': 'Permanent action threat',
    'blocked': 'Account blocked warning',
    'terminated': 'Account termination threat',
    'restore account': 'Account restoration request',
    'reactivate': 'Account reactivation request',  
    #verification
    'verify': 'Verification request',
    'confirm': 'Confirmation request',
    'update': 'Update request',
    'login': 'Login credential request',
    'password': 'Password request',
    'account recovery': 'Account recovery request',
    #suspecious claims
    'unauthorized': 'Unauthorized access claim',
    'suspicious activity': 'Suspicious activity claim',
    'unusual login': 'Unusual login detected',
    'new device detected': 'New device login alert',
    'breach': 'Security breach claim',
    'compromised': 'Compromised account claim',
    'hacked': 'Hacked account claim',
    'security': 'Security alert',
    #click request
    'click here': 'Suspicious click request',
    'click': 'Click link request',
    #fiancial
    'credit card': 'Credit card information request',
    'bank account': 'Bank account information request',
    'wire transfer': 'Wire transfer request',
    'ssn': 'SSN/personal info request',
    '$': 'Monetary amount mentioned',
    'million': 'Large monetary amount',
    'invoice': 'Invoice/financial document',
    'refund': 'Refund offer',
    'tax': 'Tax-related claim',
    'bill': 'Fake bill/invoice',
    'overdue': 'Overdue payment threat',
    'pending payment': 'Pending payment notification',
    'tax refund': 'Tax refund scam',
    #prizes 
    'winner': 'Prize/winner claim',
    'won': 'Prize claim detected',
    'prize': 'Prize offer detected',
    'reward': 'Reward offer detected',
    'free': 'Free offer detected',
    'cash': 'Cash offer detected',
    'lottery': 'Lottery scam',
    'inheritance': 'Inheritance scam',
    'investment': 'Investment scam',    
    #brand impersonation
    'irs': 'IRS/government impersonation',
    'paypal': 'PayPal impersonation',
    'amazon': 'Amazon impersonation',
    'microsoft': 'Microsoft impersonation',
    'apple': 'Apple impersonation',
    'google': 'Google impersonation',
    'netflix': 'Netflix impersonation',
    'bank': 'Bank impersonation',
    'fedex': 'FedEx impersonation',
    'ups': 'UPS impersonation',
    'dhl': 'DHL impersonation',
    'usps': 'USPS impersonation',
    'facebook': 'Facebook impersonation',
    'instagram': 'Instagram impersonation',
    'twitter': 'Twitter impersonation',
    'whatsapp': 'WhatsApp impersonation',
    'spotify': 'Streaming service impersonation',
    'prime': 'Amazon Prime impersonation',
    #tech support
    'virus detected': 'Fake virus alert',
    'malware': 'Fake malware alert',
    'tech support': 'Fake tech support',
    'remote access': 'Remote access request',
    'install software': 'Software installation request',
    'windows': 'Windows impersonation',
    #job scams
    'job offer': 'Fake job offer',
    'work from home': 'Work-from-home scam',
    'recruitment': 'Fake recruitment',
    'interview': 'Fake interview request',
    'salary': 'Fake salary offer',
    'signing bonus': 'Fake bonus offer',
    #package and delivery scam 
    'delivery': 'Fake delivery notification',
    'tracking': 'Fake tracking link',
    'shipping': 'Fake shipping notification',
    'package': 'Fake package delivery', 
    #urls
    'http': 'External URL detected',
    'www.': 'External URL detected',
    #suspecious patterns
    'free money': 'Free money scam',
    'guaranteed': 'Fake guarantee',
    'no risk': 'No risk claim',
    '100%': 'Fake percentage guarantee',
}


class PhishingModel:
    def __init__(
        self, 
        model_path='models/rf_model.joblib',
        vectorizer_path='models/tfidf.joblib'
        ):
        self.model_path      = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model           = None
        self.vectorizer      = None
        self.preprocessor    = TextPreprocessor()
        self.load_model()

    def load_model(self) -> bool:
        if not self.model_path.exists():
            print(f"Model not found: {self.model_path}")
            return False
        if not self.vectorizer_path.exists():
            print(f"Vectorizer not found: {self.vectorizer_path}")
            return False
        try:
            self.model      = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("\n ML Model loaded successfully....")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # email clening
    def preprocess_text(
        self, 
        text: str) -> str:
        return self.preprocessor.preprocess(text)

    def extract_features(
        self, 
        text: str
        ):
        if self.vectorizer is None:
            return None
        try:
            # email clean
            processed = self.preprocess_text(text)
            # vectorize nd transform.. add to list
            return self.vectorizer.transform([processed])
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def predict(
        self, 
        text: str) -> dict:
        if self.model is None or self.vectorizer is None:
            return self._fallback_prediction()

        try:
            features = self.extract_features(text)
            if features is None:
                return self._fallback_prediction()

            prediction    = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            # calculting confidence of the prediction
            confidence    = round(float(np.max(probabilities) * 100), 1)
            
            # prediction level acc to conf
            if prediction == 1:
                result = 'phishing'
                if confidence >= 85:   
                    risk_level = 'high'
                elif confidence >= 65:
                    risk_level = 'medium'
                else:                
                    risk_level = 'low'
            else:
                result     = 'safe'
                risk_level = 'low'

            risk_factors = self._extract_risk_factors(
                text, 
                prediction
                )

            return {
                'result':       result,
                'confidence':   confidence,
                'risk_level':   risk_level,
                'risk_factors': risk_factors,
                }

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction()

    def _extract_risk_factors(
        self, 
        text: str, 
        prediction: int) -> list:
        # conert to lowercase for keyword matching
        text_lower = text.lower()
        # to store unique descr
        found_kw = {}
        for keyword, description in PHISHING_KEYWORDS.items():
            # check if kw is in email
            if keyword.lower() in text_lower:
                if description not in found_kw:
                    found_kw[description] = True

        factors = list(found_kw.keys())
    
        if prediction == 1 and not factors:
            factors = ['Suspicious email pattern detected by AI model']

        if prediction == 0 and factors:
            factors = [f"{f} (monitor)" for f in factors[:2]]

        return factors[:7]

    def _fallback_prediction(self) -> dict:
        return {
            'result':       'safe',
            'confidence':   50.0,
            'risk_level':   'low',
            'risk_factors': ['Model not loaded... using fallback']
        }

    def get_model_info(self) -> dict:
        if self.model is None:
            return {
                "status": "Model not loaded",
                "using_fallback": True
                }
            
        return {
            "status":         "loaded",
            # Model ka class name nikalta hai. Jaise 'RandomForestClassifier'. 
            # type(self.model) se pata chalta hai konsi class ka object he
            "model_type":     type(self.model).__name__,
            "n_estimators":   getattr(self.model, 'n_estimators', 'unknown'),
            "features":       getattr(self.vectorizer, 'n_features', 3000),
            "using_fallback": False,
            }

# wrapper class that calls shapexplainer ... Wrapper ka matlab hota hai 
# ek middleman jo asli complex class ko simplify karta he

class SHAPExplainer:
    def __init__(
        self, 
        phishing_model: PhishingModel
        ):
        self._inner = RealShapExplainer(
            model=phishing_model.model,
            vectorizer=phishing_model.vectorizer
        )
        self.explainer = self._inner.explainer

    def explain(
        self,
        email_text: str,
        features=None) -> list:
        result = self._inner.explain(
            email_text, 
            top_n=15
            )
        return result.get('top_features', [])

if __name__ == "__main__":
    model = PhishingModel()
    if model.model is not None:
        emails = [
            "URGENT: Your PayPal account suspended! Click http://fake.com to verify now or deleted permanently!",
            "Hi, see you at the meeting tomorrow at 10 AM. Please bring the report.",
            "Congratulations! You've won $1,000,000! Claim your prize now!",
        ]
        for email in emails:
            result = model.predict(email)
            print(f"\n{email[:60]}...")
            print(f"\n   Result     : {result['result'].upper()}")
            print(f"\n   Confidence : {result['confidence']}%")
            print(f"\n   Risk       : {result['risk_level']}")
            print(f"\n   Factors    : {result['risk_factors']}\n")