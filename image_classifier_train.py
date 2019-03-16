import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import pickle
import camera_config

from feature_extraction import get_image_features
from image_classifier_evaluate import create_image_set, evaluate_model

camera = camera_config.camera

image_dir = "image data/"

# Training
print("creating train set generator")
image_generator_function, train_labels = create_image_set(os.path.join(image_dir, "train"))
print("Percent occupied examples in train: {} of {}".format(train_labels.count(True) / len(train_labels),
                                                            len(train_labels)))

gamma_range = np.linspace(0.2, 1, num=10)

# 10^2 - 10^4
C_range = np.logspace(2, 4, num=10)

param_grid = dict(gamma=gamma_range, C=C_range, )
cross_validation = KFold(5, shuffle=True, random_state=1323055)
grid_search = GridSearchCV(SVC(class_weight="balanced"), param_grid=param_grid, cv=cross_validation)

test_pipeline = Pipeline([
    ("variance", VarianceThreshold()),
    ("scaler", MinMaxScaler()),
    ("grid_search", grid_search)])
print("loading training set")
features = [np.hstack(get_image_features(image)) for image in image_generator_function()]
print("Begin Training")
clf_model = test_pipeline
clf_model.fit(features, train_labels)

pickle.dump(clf_model, open(camera_config.classifier_model_filename, 'wb'))

print("Training Confusion Mat:\n {}".format(
    confusion_matrix(train_labels, clf_model.predict(features))))

evaluate_model(clf_model)
