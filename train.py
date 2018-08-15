import pandas as pd
import xgboost as xgb
import pickle
import csv

final_train_data = pd.read_csv('final_train.csv')
final_test_data = pd.read_csv('final_test.csv')

n_building = final_train_data.shape[0]

n_features = final_train_data.shape[1] - 1

X_all = final_train_data.drop(['damage_grade', 'building_id'], 1)

y_all = final_train_data['damage_grade']

cols = list(X_all.columns.values)

from sklearn.preprocessing import scale

for col in cols:
  X_all[col] = scale(X_all[col])

def preprocess_features(X):
  output = pd.DataFrame(index=X.index)

  for col, col_data in X.items():
    if col_data.dtype == object:
      col_data = pd.get_dummies(col_data, prefix=col)
    output = output.join(col_data)
  return output

X_all = preprocess_features(X_all)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all,
                                    y_all,
                                    train_size=80,
                                    test_size=20,
                                    random_state=2,
                                    stratify=y_all)


from time import time
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
  print ('started training model')
  start = time()
  clf.fit(X_train, y_train)
  end = time()
  pickle.dump(clf, open('trained-model.dat', 'wb'))
  print ('Trained model in {:.4f} seconds'.format(end-start))


def predict_labels(clf, features, target):
  print ('predicting labels')
  start = time()
  y_pred = clf.predict(features)
  end = time()

  print ('Made Predictions in {:.4f} seconds'.format(end-start))

  return f1_score(target, y_pred, average=None), sum(target==y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
  train_classifier(clf, X_train, y_train)

  f1, acc = predict_labels(clf, X_train, y_train)
  print (f1, acc)
  f1, acc = predict_labels(clf, X_test, y_test)
  print (f1, acc)


clf_C = xgb.XGBClassifier(seed=82)

train_predict(clf_C, X_train, y_train, X_test, y_test)
