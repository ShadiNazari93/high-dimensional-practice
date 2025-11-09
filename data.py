from ucimlrepo import fetch_ucirepo
# fetch dataset
myocardial_infarction_complications = fetch_ucirepo(id=579)
# data (as pandas dataframes)
X = myocardial_infarction_complications.data.features
y = myocardial_infarction_complications.data.targets
# metadata
print(myocardial_infarction_complications.metadata)
# variable information
print(myocardial_infarction_complications.variables)
print(X)