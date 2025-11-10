import joblib
def save_model(clf, path="svm_model.joblib"):
    joblib.dump(clf, path)
    print(f"Model saved to {path}")

def load_model(path="svm_model.joblib"):
    clf = joblib.load(path)
    print(f"Model loaded from {path}")
    return clf