import pandas as pd


train_data = pd.read_csv('CleanedDataset/train.csv')

test_data = pd.read_csv('CleanedDataset/test.csv')

building_structure_data = pd.read_csv('CleanedDataset/Building_Structure.csv')
building_ownership_data = pd.read_csv('CleanedDataset/Building_Ownership_Use.csv')


building_structure_ownership_merged_data = pd.merge(building_structure_data,
                      building_ownership_data,
                      how='inner',
                      left_on=['building_id'],
                      right_on=['building_id'])

final_train_data = pd.merge(train_data,
                      building_structure_ownership_merged_data,
                      how='inner',
                      left_on=['building_id'],
                      right_on=['building_id'])
final_train_data.to_csv('final_train.csv',index=False)

final_test_data = pd.merge(test_data,
                      building_structure_ownership_merged_data,
                      how='inner',
                      left_on=['building_id'],
                      right_on=['building_id'])
final_test_data.to_csv('final_test.csv', index=False)
