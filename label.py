import pandas as pd
import pickle
import csv
from time import time

with open('FinalDataset/trained-model.dat', 'rb') as model_f:
  model = pickle.load(model_f)

test_data = pd.read_csv('FinalDataset/final_test.csv')
building_ids = list(test_data['building_id'].values)

test_data = test_data.drop(['building_id'], 1)

cols = list(test_data.columns.values)

from sklearn.preprocessing import scale

for col in cols:
  test_data[col] = scale(test_data[col])


def preprocess_features(X):
  output = pd.DataFrame(index=X.index)

  for col, col_data in X.items():
    if col_data.dtype == object:
      col_data = pd.get_dummies(col_data, prefix=col)
    output = output.join(col_data)
  return output

test_data = preprocess_features(test_data)

def predict_labels(features):
  print ('predicting labels')
  start = time()
  y_pred = model.predict(features)
  end = time()
  print ('Made Predictions in {:.4f} seconds'.format(end-start))
  return y_pred

predict_result = predict_labels(test_data)

with open('FinalDataset/result.csv', 'w') as output_f:
  csvfw = csv.writer(output_f, dialect='excel')
  data = [[building_ids[x],predict_result[x]] for x in range(len(predict_result))]
  csvfw.writerows([['building_id', 'damage_grade']] + data)
