from sklearn.svm import LinearSVC
model_path = "svm_model.joblib"

def svm(X_train, y_train,C = 1.0):
    clf = LinearSVC(
        C=C,
        class_weight='balanced',  # 如果类别不均衡，自动给少数类更高权重
        max_iter=10000,
        dual=False  # 对样本数 >> 特征数很有用；你可以试试 True/False 哪个更好
    )
    print("Start training SVM...")
    clf.fit(X_train, y_train)
    print("SVM training finished.")

    return clf