from feature_extraction import *
from classifier import *
from util.model import save_model, load_model
csvfile = "train_small.csv"
clf_path = "svm_model.joblib"
get_patches(csvfile)
bow_extraction()
x, y = get_train_data()

clf = svm(x,y)
save_model(clf,clf_path)

