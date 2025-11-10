from sklearn.svm import LinearSVC
model_path = "svm_model.joblib"

def svm(X_train, y_train,C = 1.0):
    clf = LinearSVC(
        C=C,
        class_weight='balanced',
        max_iter=10000,
        dual=False
    )
    print(X_train, y_train)
    print("Start training SVM...")
    clf.fit(X_train, y_train)
    print("SVM training finished.")

    return clf