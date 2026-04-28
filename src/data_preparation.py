from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
dataset_path = Path('C:/Python/Cancer_p1/data/cancer_reg.csv')
df = pd.read_csv(dataset_path)
df.drop(['Geography', 'binnedInc'], axis=1, inplace=True, errors='ignore')
df.fillna(df.median(), inplace=True)

selected_features = [
    'PctPublicCoverageAlone', 'incidenceRate', 'povertyPercent', 'PctHS25_Over', 'PctPublicCoverage',
    'PctEmployed16_Over', 'PctHS18_24', 'PctBlack', 'PctNoHS18_24', 'MedianAgeFemale', 'MedianAge', 'studyPerCap',
    'MedianAgeMale','AvgHouseholdSize','avgDeathsPerYear','BirthRate','popEst2015','avgAnnCount','PctWhite','PctAsian',
    'PctOtherRace','PctEmpPrivCoverage','PercentMarried','PctBachDeg18_24','PctMarriedHouseholds','PctPrivateCoverage',
    'PercentMarried','PctBachDeg18_24','PctMarriedHouseholds','PctPrivateCoverage','PctEmployed16_Over','medIncome',
    'PctBachDeg25_Over'
]
selected_features2 = [
    'PctPublicCoverageAlone', 'incidenceRate', 'povertyPercent', 'PctHS25_Over', 'PctPublicCoverage',
    'PctEmployed16_Over', 'PctHS18_24', 'PctBlack', 'PctNoHS18_24', 'MedianAgeFemale', 'MedianAge']
selected_features3 = ['PctPublicCoverageAlone', 'incidenceRate', 'povertyPercent', 'PctHS25_Over', 'PctPublicCoverage',
    'PctEmployed16_Over', 'PctHS18_24', 'PctBlack']
target_col = 'TARGET_deathRate'

# , 'PctNoHS18_24', 'MedianAgeFemale', 'MedianAge', 'studyPerCap',
#     'MedianAgeMale','AvgHouseholdSize','avgDeathsPerYear','BirthRate','popEst2015','avgAnnCount','PctWhite','PctAsian',
#     'PctOtherRace','PctEmpPrivCoverage','PercentMarried','PctBachDeg18_24','PctMarriedHouseholds','PctPrivateCoverage',
#     'PercentMarried','PctBachDeg18_24','PctMarriedHouseholds','PctPrivateCoverage','PctEmployed16_Over','medIncome',
#     'PctBachDeg25_Over'

X = df[selected_features3]
y = df[target_col]

seed = None

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=seed
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

y_train_np = y_train.values
y_valid_np = y_valid.values
y_test_np = y_test.values

np.save('C:/Python/Cancer_p1/data/processed/X_train.npy', X_train_scaled)
np.save('C:/Python/Cancer_p1/data/processed/X_valid.npy', X_valid_scaled)
np.save('C:/Python/Cancer_p1/data/processed/X_test.npy', X_test_scaled)
np.save('C:/Python/Cancer_p1/data/processed/y_train.npy', y_train_np)
np.save('C:/Python/Cancer_p1/data/processed/y_valid.npy', y_valid_np)
np.save('C:/Python/Cancer_p1/data/processed/y_test.npy', y_test_np)

pd.DataFrame(X_train_scaled, columns=selected_features3).to_csv('C:/Python/Cancer_p1/data/processed/X_train.csv', index=False)
pd.DataFrame(X_valid_scaled, columns=selected_features3).to_csv('C:/Python/Cancer_p1/data/processed/X_valid.csv', index=False)
pd.DataFrame(X_test_scaled, columns=selected_features3).to_csv('C:/Python/Cancer_p1/data/processed/X_test.csv', index=False)
y_train.to_csv('C:/Python/Cancer_p1/data/processed/y_train.csv', index=False)
y_valid.to_csv('C:/Python/Cancer_p1/data/processed/y_valid.csv', index=False)
y_test.to_csv('C:/Python/Cancer_p1/data/processed/y_test.csv', index=False)

print("Предобработка завершена.")
print(f"Train: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
print(f"Valid: {X_valid_scaled.shape[0]} samples")
print(f"Test: {X_test_scaled.shape[0]} samples")
print("Файлы сохранены: X_train.npy, X_valid.npy, X_test.npy, y_train.npy, y_valid.npy, y_test.npy, а также CSV-копии.")