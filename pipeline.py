import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv("C:\\Users\\acer\\Downloads\\10 pipe dataset.csv")
# print(df.head())
# print( X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# print(df["Accident_severity"].value_counts().plot(kind= "bar"))
# print(df["Educational_level"].value_counts())
df["Time"] = pd.to_datetime(df["Time"])
df["Hour_of_Day"] = df["Time"].dt.hour
# print(df["Hour_of_Day"])
new_df = df.copy()
new_df.drop("Time",axis=1,inplace=True) 
le = LabelEncoder()
new_df["Accident_severity"]= le.fit_transform( new_df["Accident_severity"])
# print(new_df["Accident_severity"].value_counts())
# print("Before Sampling")
X = new_df.drop("Accident_severity",axis=1)
y = new_df["Accident_severity"]
random_smpl = RandomOverSampler( random_state=42)
X_reassemple,y_reassemple = random_smpl.fit_resample( X,y)
# print(y_reassemple.value_counts())
X_train, X_test, y_train, y_test = train_test_split( X_reassemple,y_reassemple, test_size=0.2, random_state=42)
# print( X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# print(X_test.info())

# print(new_df.info())

#                              fill missing values

trf1 = ColumnTransformer(
    transformers=[
        ("impute_Educational_level",SimpleImputer(strategy="most_frequent"),[3]),
        ("impute_Vehicle_driver_relation", SimpleImputer(strategy="most_frequent"),[4]),
        ("impute_Driving_experience", SimpleImputer(strategy="most_frequent"),[5]),
        ("impute_Type_of_vehicle", SimpleImputer(strategy="most_frequent"),[6]),
        ("impute_Owner_of_vehicle", SimpleImputer(strategy="most_frequent"),[7]),
        ("impute_Service_year_of_vehicle", SimpleImputer(strategy="most_frequent"),[8]),
        ("impute_Defect_of_vehicle", SimpleImputer(strategy="most_frequent"),[9]),
        ("impute_Area_accident_occured", SimpleImputer(strategy="most_frequent"),[10]),
        ("impute_Lanes_or_Medians", SimpleImputer(strategy="most_frequent"),[11]),
        ("impute_Road_allignment", SimpleImputer(strategy="most_frequent"),[12]),
        ("impute_Types_of_Junction", SimpleImputer(strategy="most_frequent"),[13]),
        ("impute_Road_surface_type", SimpleImputer(strategy="most_frequent"),[14]),
        ("impute_Type_of_collision", SimpleImputer(strategy="most_frequent"),[18]),
        ("impute_Vehicle_movement", SimpleImputer(strategy="most_frequent"),[21]),
        ("impute_Work_of_casuality", SimpleImputer(strategy="most_frequent"),[26]),
        ("impute_Fitness_of_casuality", SimpleImputer(strategy="most_frequent"),[27]),
    ],
    remainder='passthrough',
)

#                         Encode categorical columns
obj_columns_indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

trf2 = ColumnTransformer(
    transformers=[
        (f'ohe_{col}', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [col]) 
        for col in obj_columns_indexes
    ],
    remainder='passthrough',
)
X_train_encoded = trf2.fit_transform(X_train)
# print(X_train_encoded.shape)

#              select top k features using mutual information

trf3 = SelectKBest(chi2,k=50)
#               train model

trf4=RandomForestClassifier()
 #            craete pipeline 

pipe = Pipeline([
    ('impute', trf1),
    ('ohe', trf2),
    ('select', trf3),
    ('rf', trf4),
])
pipe.fit(X_train, y_train)
# print(pipe.get_feature_names_out)

#                     check accuracy of model on test data

y_pred = pipe.predict(X_test)
acc_score = accuracy_score( y_test, y_pred)
# print(acc_score)
clf_report = classification_report(y_test,y_pred)
# print(clf_report)
cnf_matrix = confusion_matrix( y_test, y_pred)
# print(cnf_matrix)

pickle.dump( pipe, open( "pipeline.pkl","wb" ) )

data = [
    "Day_of_week","Age_band_of_driver","Sex_of_driver","Educational_level",
    "Vehicle_driver_relation","Driving_experience","Type_of_vehicle","Owner_of_vehicle",
    "Service_year_of_vehicle","Defect_of_vehicle","Area_accident_occured","Lanes_or_Medians",
    "Road_allignment","Types_of_Junction","Road_surface_type","Road_surface_conditions",
    "Light_conditions","Weather_conditions","Type_of_collision","Number_of_vehicles_involved",
    "Number_of_casualties","Vehicle_movement","Casualty_class","Sex_of_casualty",
    "Age_band_of_casualty","Casualty_severity","Work_of_casuality","Fitness_of_casuality",
    "Pedestrian_movement","Cause_of_accident","Hour_of_Day"
]

# Example single input
data2 = [
    "Thursday", "31-50", "Male", "Junior high school", "Owner", "5-10yrs",
    "Long lorry", "Owner", "Unknown", "Other", "Two-way (divided with solid lines road marking)",
    "Tangent road with flat terrain", "Crossing", "Y Shape", "Dry", "Daylight",
    "Normal", "Collision with animals", 2, 1, "Going straight", "Driver or rider",
    "Male", "18-30", 3, "Driver", "Normal", "Not a Pedestrian", "Changing lane to the left",
    "Overtaking", 12
]

def predict(data_input):
    features = np.array(data_input).reshape(1, -1)
    results = pipe.predict(features)
    return results

predicted_class = predict(data2)
if predicted_class[0] == 2:
    print("Slight Injury")
elif predicted_class[0] == 1:
    print("Serious Injury")
else:
    print("Fatal Injury")
# print("Predicted Accident Severity:", predicted_class)

