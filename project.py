from feature_extraction import *
from classifier import *
from util.model import save_model, load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test(clf, patches):
    # 在验证集上预测
    X_val, y_val = get_train_data(patches)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print("Validation accuracy:", acc)

    print(classification_report(y_val, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

csvfile = "train_small.csv"
clf_path = "svm_model.joblib"

# # Train
# train_patches = get_patches(csvfile,10000)
# bow_extraction(train_patches)
# x, y = get_train_data(train_patches)

# Save model
# clf = svm(x,y)
# save_model(clf,clf_path)

# smoke test
csvfile = "valid.csv"
valid_patches = get_patches(csvfile, 5000)
bow_extraction(valid_patches)
st_clf = load_model(clf_path)
print("Start testing SVM...")
test(st_clf,valid_patches)
print("Finished testing SVM...")
