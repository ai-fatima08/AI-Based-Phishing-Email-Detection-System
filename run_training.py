import os
import sys
import time
from sklearn.model_selection import train_test_split

from dataset_load   import DatasetLoader
from model_train    import ModelTrainer
from shap_explainer import SHAPExplainer

DATASET_PATHS = [
    r"C:\Users\fiaaa\Desktop\phishing_detection_system\datasets\CEAS_08.csv",
    r"C:\Users\fiaaa\Desktop\phishing_detection_system\datasets\Phishing_Email.csv",
    r"C:\Users\fiaaa\Desktop\phishing_detection_system\datasets\phishing_emails.csv",
    ]

MODEL_SAVE_DIR     = "models"
TFIDF_MAX_FEATURES = 5000
RF_N_ESTIMATORS    = 300
RF_MAX_DEPTH       = 20
TEST_SIZE          = 0.2
RANDOM_STATE       = 42
MAX_TRAIN_SAMPLES  = 30000   

def run():
    total_start = time.time()

    print("\n" + "*" * 110)
    print("  PHISHING DETECTION TRAINING ")
    print("*" * 110)

    found   = [p for p in DATASET_PATHS if os.path.exists(p)]
    missing = [p for p in DATASET_PATHS if not os.path.exists(p)]

    if missing:
        print("\n These files are missing:")
        for p in missing:
            print(f"    {p}")
    if not found:
        print("\n No Dataset Found...")
        sys.exit(1)

    print("\n ------------------------ Datasets load + preprocess ------------------------ ")
    loader = DatasetLoader(
        dataset_paths=DATASET_PATHS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        min_text_length=10
    )
    X_train, X_test, y_train, y_test = loader.load_and_split()

    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(f"\n RAM limit: {len(X_train):,} → {MAX_TRAIN_SAMPLES:,} samples (stratified)")
        X_train, _, y_train, _ = train_test_split(
            X_train, 
            y_train,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
        print(f"Training set reduced: {len(X_train):,} samples "
              f"(P:{(y_train==1).sum():,}  S:{(y_train==0).sum():,})")

    print("\n  ------------------------ Model training ------------------------ ")
    trainer = ModelTrainer(
        save_model=MODEL_SAVE_DIR,
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        tfidf_max_features=TFIDF_MAX_FEATURES,
        random_state=RANDOM_STATE
    )
    metrics = trainer.train(X_train, X_test, y_train, y_test)
    trainer.save()

    print("\n  ------------------------ SHAP ------------------------ ")
    shap_exp = SHAPExplainer(
        model=trainer.model, 
        vectorizer=trainer.vectorizer
    )
    result = shap_exp.explain("URGENT: Your account SUSPENDED! Verify now.", top_n=7)
    print(f"    Using SHAP : {result['using_shap']}")
    print(f"    Summary    : {result['summary']}")

    total_time = time.time() - total_start

    print("\n" + "*" * 110)
    print("  TRAINING COMPLETED SUCCESSFULLY...")
    print("*" * 110)
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  Time      : {total_time:.1f}s")
    print(f"  Saved     : {MODEL_SAVE_DIR}/rf_model.joblib  +  tfidf.joblib")
    print("*" * 110)
    return metrics

if __name__ == "__main__":
    run()