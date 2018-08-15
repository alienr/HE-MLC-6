import csv

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

with open('Dataset/train.csv') as train_f:
  train_csv = csv.reader(train_f, dialect='excel')
  train_csv_header = next(train_csv)
  train_csv_data = [_ for _ in train_csv]
for row in train_csv_data:
  if row[12] == '':
    row[12] = 0


with open('Dataset/test.csv') as test_f:
  test_csv = csv.reader(test_f, dialect='excel')
  test_csv_header = next(test_csv)
  test_csv_data = [_ for _ in test_csv]
for row in test_csv_data:
  if row[11] == '':
    row[11] = 0

area_assesed_col_values = {row[0] for row in train_csv_data}
area_assesed_col_values2 = {row[0] for row in test_csv_data}

encoded_values = le.fit(list(area_assesed_col_values))
labels = encoded_values.classes_
transformed = le.transform(list(area_assesed_col_values))

area_assesed_mapping = {labels[i]: transformed[i] for i in range(len(labels))}

train_csv_f = open('CleanedDataset/train.csv', 'w')
test_csv_f = open('CleanedDataset/test.csv', 'w')

cleaned_train_csv = csv.writer(train_csv_f, dialect='excel')
cleaned_test_csv = csv.writer(test_csv_f, dialect='excel')

cleaned_train_csv_data = [train_csv_header]
cleaned_test_csv_data = [test_csv_header]

for row in train_csv_data:
  row[0] = area_assesed_mapping[row[0]]
  cleaned_train_csv_data.append(row)

for row in test_csv_data:
  row[0] = area_assesed_mapping[row[0]]
  cleaned_test_csv_data.append(row)

cleaned_train_csv.writerows(cleaned_train_csv_data)
cleaned_test_csv.writerows(cleaned_test_csv_data)

train_csv_f.close()
test_csv_f.close()

with open('Dataset/Building_Ownership_Use.csv') as ownership_f:
  ownership_csv = csv.reader(ownership_f, dialect='excel')
  ownership_csv_header = next(ownership_csv)
  ownership_csv_data = [_ for _ in ownership_csv]
for row in ownership_csv_data:
  for idx, col in enumerate(row):
    if col == '':
      row[idx] = 0.0

legal_ownership_status_col_values = {row[4] for row in ownership_csv_data}

encoded_values = le.fit(list(legal_ownership_status_col_values))
labels = encoded_values.classes_
transformed = le.transform(list(legal_ownership_status_col_values))
legal_ownership_status_mapping = {labels[i]: transformed[i] for i in range(len(labels))}

cleaned_ownership_use_f = open('CleanedDataset/Building_Ownership_Use.csv', 'w')

cleaned_ownership_use_csv = csv.writer(cleaned_ownership_use_f, dialect='excel')

cleaned_ownership_use_csv_data = [ownership_csv_header]

for row in ownership_csv_data:
  row[4] = legal_ownership_status_mapping[row[4]]
  cleaned_ownership_use_csv_data.append(row)

cleaned_ownership_use_csv.writerows(cleaned_ownership_use_csv_data)

cleaned_ownership_use_f.close()

structure_csv = csv.reader(open('Dataset/Building_Structure.csv'), dialect='excel')

structure_csv_header = next(structure_csv)

structure_csv_data = [_ for _ in structure_csv]
for row in structure_csv_data:
  for idx, col in enumerate(row):
    if col == '':
      if idx == 15:
        row[idx] = 'Attached-1 side'
      elif idx == 16:
        row[idx] = 'Rectangular'

encode_cols = ['area_assesed'] # train csv
encode_cols2 = ['legal_ownership_status'] # Building Ownership User
encode_cols3 = ['land_surface_condition', # Building Structure
                 'foundation_type',
                 'roof_type',
                 'ground_floor_type',
                 'other_floor_type',
                 'position',
                 'plan_configuration',
                 'condition_post_eq']
encode_cols4 = ['area_assesed'] # test csv

col_index = {col: structure_csv_header.index(col) for col in encode_cols3}
col_values = {col: set() for col in encode_cols3}
col_value_mapping = {}
for row in structure_csv_data:
  for col in encode_cols3:
    col_values[col].add(row[col_index[col]])

for k, v in col_values.items():
  encoded_values = le.fit(list(v))
  labels = encoded_values.classes_
  transformed = le.transform(list(v))
  v_mapping = {labels[i]: transformed[i] for i in range(len(labels))}
  col_value_mapping[k]= v_mapping

structure_csv_f = open('CleanedDataset/Building_Structure.csv', 'w')

cleaned_structure_csv = csv.writer(structure_csv_f, dialect='excel')

cleaned_structure_csv_data = [structure_csv_header]

for row in structure_csv_data:
  for col in encode_cols3:
    row[col_index[col]] = col_value_mapping[col][row[col_index[col]]]
  cleaned_structure_csv_data.append(row)

cleaned_structure_csv.writerows(cleaned_structure_csv_data)

structure_csv_f.close()
