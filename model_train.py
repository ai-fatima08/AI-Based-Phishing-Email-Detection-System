import gc
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("imblearn not installed. Run: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False


class ModelTrainer:
    def __init__(self, save_model="models",
                 n_estimators=500,
                 max_depth=20,
                 tfidf_max_features=5000,
                 random_state=42
                 ):
        self.save_model         = Path(save_model)
        self.n_estimators       = n_estimators
        self.max_depth          = max_depth
        self.tfidf_max_features = tfidf_max_features
        self.random_state       = random_state
        self.vectorizer         = None
        self.model              = None
        self.metrics            = {}
        self.model_path         = self.save_model / "rf_model.joblib"
        self.vectorizer_path    = self.save_model / "tfidf.joblib"

# x--features , y--labels

    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test
        ):
        print(
            "\n" + "*" * 110
            )
        print(
            " MODEL TRAINING Starting..."
            )
        print(
            "*" * 110
            )
        # converting features into numbers
        X_train_vec, X_test_vec = self._vectorize(X_train, X_test)
        # fixing class imb
        X_balanced, y_balanced  = self._apply_smote(X_train_vec, y_train)
        # Free memory after SMOTE
        del X_train_vec
        gc.collect()
        # call _train_rf... training rf model on balnced dt
        self._train_rf(
            X_balanced, 
            y_balanced
            )
        # call _evaluate... model evaluate on testing data
        self.metrics = self._evaluate(
            X_test_vec, 
            y_test
            )
        print(
            "\n Training completed successfully..."
            )
        return self.metrics

    # save trained model and vectorizer file
    def save(self):
        # creating path for saving file
        self.save_model.mkdir(parents=True, exist_ok=True)
        if self.model is None or self.vectorizer is None:
            print("First Call the train()")
            return False

        # dump... Joblib library ka function jo object save karta he
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

        # stat... metadata of file, st_size... file size into bytes then conert into mb
        model_mb = self.model_path.stat().st_size / (1024 * 1024)
        vec_mb   = self.vectorizer_path.stat().st_size / (1024 * 1024)

        print(
            f"\n------------------- Model saved Successfully -------------------"
            )
        # file path and size, 2f... 2 decimal places
        print(f"   {self.model_path}  ({model_mb:.2f} MB)")
        print(f"   {self.vectorizer_path}  ({vec_mb:.2f} MB)")
        return True

    # ye rf k imp features nikalta he  
    def get_feature_importance(
        self, 
        top_n=30   #30 top featurs return kryga
        ):
        if self.model is None:
            return []
        # getting feature imp vals
        importances = self.model.feature_importances_
        # imp k acc ind ko sort kro asc order, top n.. last n high imp val
        # ::1.. reverse so that top imp feature pehle aye 
        top_idx     = np.argsort(importances)[-top_n:][::-1]
        return [
            {
                'feature': f'feat_{i}', 'importance': round(float(importances[i]), 5)
                }
            for i in top_idx
            ]

    # HashingVectorizer.. convert text to numb
    def _vectorize(
        self,
        X_train, 
        X_test
        ):
        print(
            f"\n Vectorization (n_features={self.tfidf_max_features})..."
            )
        # obj to convert text into numbers
        self.vectorizer = HashingVectorizer(
            n_features     = self.tfidf_max_features,
            ngram_range    = (1, 1),       # consider only single words
            alternate_sign = False,        # only positive values 
            norm           = 'l2',         # normailze the number
            dtype          = np.float32
        )
        gc.collect()
        X_train_vec = self.vectorizer.transform(X_train)
        gc.collect()
        X_test_vec  = self.vectorizer.transform(X_test)
        gc.collect()

        # shape... kitni emails or kitny feautres 
        print(
            f"Train: {X_train_vec.shape[0]:,}  {X_train_vec.shape[1]:,}"
            )
        print(
            f"Test : {X_test_vec.shape[0]:,}  {X_test_vec.shape[1]:,}"
            )
        return X_train_vec, X_test_vec

    # SMOTE
    def _apply_smote(
        self, 
        X_train_vec, 
        y_train
        ):
        before = Counter(y_train)   #count labels
        print(
            f"\n Class distribution: {dict(before)}"
            )
        if not SMOTE_AVAILABLE:
            print("SMOTE not available....")
            return X_train_vec, y_train

        minority = min(before.values())
        majority = max(before.values())

        # ratio check.. 
        if majority / minority < 1.5:
            print(
                "Classes already balanced..."
                )
            return X_train_vec, y_train
        
        # applying smote.. create synthetic samples
        smote = SMOTE(
            random_state=self.random_state
            )
        X_balanced, y_balanced = smote.fit_resample(X_train_vec, y_train)
        print(
            f"After SMOTE: {dict(Counter(y_balanced))}"
            )
        return X_balanced, y_balanced

     # RANDOM FOREST
    def _train_rf(
        self, 
        X_balanced,
        y_balanced
        ):
        print(
            f"\n------------------- Training Random Forest -------------------"
            f"(\n={self.n_estimators}, depth={self.max_depth})..."
            )
        self.model = RandomForestClassifier(
            n_estimators      = self.n_estimators,
            max_depth         = self.max_depth,
            random_state      = self.random_state,
            class_weight      = 'balanced' if not SMOTE_AVAILABLE else None,
            n_jobs            = -1,
            min_samples_split = 5,
            min_samples_leaf  = 2
        )
        self.model.fit(X_balanced, y_balanced)
        print(
            "Model training completed successfully"
            )

    # EVALUATION
    def _evaluate(
        self, 
        X_test_vec,
        y_test
        ):
        print("\n---------------------------- Evaluating ----------------------------")
        # make prediction on test emails
        y_pred   = self.model.predict(X_test_vec)
        # right pred/total pred
        accuracy = accuracy_score(y_test, y_pred)
        # f1.. Precision aur recall ka harmonic mean
        f1       = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{'*'*110}")
        print(f"RESULTS")
        print(f"{'*'*110}")
        
        print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  F1-Score : {f1:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Safe', 'Phishing'])}")

        # cm func class k liye 2*2 ka matrix bnta he
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"--------------------------- Confusion Matrix ---------------------------")
        print(f"  Safe   Phish")
        
        # :5d... 5 spaces width mein integer print karo (right-aligned)
        print(f"  Actual Safe   {tn:5d}   {fp:5d}")
        print(f"  Actual Phish  {fn:5d}   {tp:5d}")
        print(f"\n  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")

        return {
            'accuracy':  round(accuracy, 4),
            'f1_score':  round(f1, 4),
            'true_pos':  int(tp),
            'true_neg':  int(tn),
            'false_pos': int(fp),
            'false_neg': int(fn),
        }

#checking
if __name__ == "__main__":
    from dataset_load import DatasetLoader

    PATHS = [
        r"C:\Users\fiaaa\Desktop\phishing_detection_system\datasets\CEAS_08.csv",
        r"C:\Users\fiaaa\Desktop\phishing_detection_system\datasets\Phishing_Email.csv",
        r"C:\Users\fiaaa\Desktop\phishing_detection_system\datasets\phishing_emails.csv",
        ]

    loader  = DatasetLoader(dataset_paths=PATHS)
    # data load + preprocess + split karo
    X_train, X_test, y_train, y_test = loader.load_and_split()

    trainer = ModelTrainer(save_model="models")
    metrics = trainer.train(X_train, X_test, y_train, y_test)
    trainer.save()

    print(f"\n Everything Okay.... Accuracy: {metrics['accuracy']*100:.2f}%")