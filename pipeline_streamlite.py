from PIL import Image
import numpy as np
import streamlit as st
import pickle 

pipe =pickle.load( open("pipeline.pkl", "rb"))
img = Image.open("D:\\Scikit-learn\\acc_img.png", 'r')
# img = img.resize((224, 224))

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


def predict(data_input):
    features = np.array(data_input).reshape(1, -1)
    results = pipe.predict(features)
    return results

st.title("Accident Prediction with Pipeline")
st.image( img, use_container_width= True)

with st.sidebar :
    st.write(" Select the features to predict")
    Day_of_week = st.selectbox('Day of Week',
                               ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Age_band_of_driver = st.selectbox('Age Band of Driver',
                                      ['17-20', '21-25', '26-30', '31-35', '36-40', '41-45',
                                       '46-50', '51-55', '56-65', '66-75', 'Over 75'])
    Sex_of_driver = st.selectbox('Sex of Driver', ['Male', 'Female', 'Unknown'])
    Educational_level = st.selectbox('Educational Level',
                                      ['Above high school', 'Junior high school', 'Elementary school',
                                       'High school', 'Unknown', 'Illiterate', 'Writing & reading'])
    Vehicle_driver_relation = st.selectbox('Vehicle Driver Relation', ['Employee', 'Unknown', 'Owner', 'Other'])

    Driving_experience = st.selectbox('Driving Experience',
                                  ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence',
                                   'Below 1yr', 'unknown'])

    Type_of_vehicle = st.selectbox('Type of Vehicle',
                               ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Public (13 seats)',
                                'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi',
                                'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle',
                                'Special vehicle', 'Bicycle'])

    Owner_of_vehicle = st.selectbox('Owner of Vehicle', ['Owner', 'Governmental', 'Organization', 'Other'])

    Service_year_of_vehicle = st.selectbox('Service Year of Vehicle',
                                       ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown'])

    Defect_of_vehicle = st.selectbox('Defect of Vehicle', ['No defect', '7', '5'])

    Area_accident_occured = st.selectbox('Area of Accident Occurred',
                                     ['Residential areas', 'Office areas', 'Recreational areas',
                                      'Industrial areas', 'Other', 'Church areas', 'Market areas',
                                      'Unknown', 'Rural village areas', 'Outside rural areas',
                                      'Hospital areas', 'School areas', 'Rural village areasOffice areas',
                                      'Recreational areas'])

    Lanes_or_Medians = st.selectbox('Lanes or Medians',
                                ['Undivided Two way', 'other', 'Double carriageway (median)',
                                 'Two-way (divided with solid lines road marking)',
                                 'Two-way (divided with broken lines road marking)', 'Unknown'])
    Road_alignment = st.selectbox('Road Alignment',
                              ['Tangent road with flat terrain', 'Tangent road with mild grade and',
                               'Escarpments', 'Tangent road with rolling terrain', 'Gentle horizontal curve',
                               'Tangent road with mountainous terrain and', 'Steep grade downward',
                               'Sharp reverse curve', 'Steep grade upward with mountainous terrain'])

    Types_of_Junction = st.selectbox('Types of Junction',
                                 ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'X Shape'])

    Road_surface_type = st.selectbox('Road Surface Type',
                                 ['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress',
                                  'Gravel roads', 'Other'])

    Road_surface_conditions = st.selectbox('Road Surface Conditions',
                                       ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm', 'Other'])

    Light_conditions = st.selectbox('Light Conditions',
                                ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                                 'Darkness - lights unlit'])

    Weather_conditions = st.selectbox('Weather Conditions',
                                  ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other',
                                   'Snow', 'Unknown', 'Fog or mist'])

    Type_of_collision = st.selectbox('Type of Collision',
                                 ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
                                  'Collision with roadside objects', 'Collision with animals', 'Rollover',
                                  'Fall from vehicles', 'Collision with pedestrians',
                                  'With Train', 'Unknown'])
    Number_of_vehicles_involved = st.number_input('Number of Vehicles Involved',
                                              min_value=1, max_value=10, step=1, value=1)

    Number_of_casualties = st.number_input('Number of Casualties',
                                       min_value=1, max_value=10, step=1, value=1)

    Vehicle_movement = st.selectbox('Vehicle Movement',
                                ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go',
                                 'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping',
                                 'Overtaking', 'Other', 'Entering a junction'])
    Casualty_class = st.selectbox('Casualty Class',
                              ['na', 'Driver or rider', 'Pedestrian', 'Passenger'])

    Sex_of_casualty = st.selectbox('Sex of Casualty',
                               ['na', 'Male', 'Female'])

    Age_band_of_casualty = st.selectbox('Age Band of Casualty',
                                    ['na', '31-50', '18-30', 'Under 18', 'Over 51', '5'])

    Casualty_severity = st.selectbox('Casualty Severity',
                                 ['na', '3', '2', '1'])

    Work_of_casualty = st.selectbox('Work of Casualty',
                                ['Driver', 'Other', 'Unemployed', 'Employee', 'Self-employed'])

    Fitness_of_casualty = st.selectbox('Fitness of Casualty',
                                   ['Normal', 'Deaf', 'Other', 'Blind', 'NormalNormal'])

    Pedestrian_movement = st.selectbox('Pedestrian Movement',
                                   ["Not a Pedestrian",
                                    "Crossing from driver's nearside",
                                    "Crossing from nearside - masked by parked or stationary vehicle",
                                    "Unknown or other",
                                    "Crossing from offside - masked by parked or stationary vehicle",
                                    "In carriageway, stationary - not crossing",
                                    "Walking along in carriageway, back to traffic",
                                    "Walking along in carriageway, facing traffic",
                                    "In carriageway, stationary - not crossing"])

    Cause_of_accident = st.selectbox('Cause of Accident',
                                 ['Moving Backward', 'Overtaking', 'Changing lane to the left',
                                  'Changing lane to the right', 'Overloading', 'Other',
                                  'No priority to vehicle', 'No priority to pedestrian',
                                  'No distancing', 'Getting off the vehicle improperly',
                                  'Improper parking', 'Overspeed', 'Driving carelessly',
                                  'Driving at high speed', 'Driving to the Left', 'Unknown',
                                  'Overturning', 'Turnover', 'Driving under the influence of drugs',
                                  'Drunk driving'])

    Hour_of_Day = st.selectbox('Hour of Day',
                           [17, 1, 14, 22, 8, 15, 12, 18, 13, 20, 16, 21, 9, 10, 19, 11, 23, 7, 0, 5])

if st.sidebar.button("Predict"):
    # Arrange inputs in the same order as 'data'
    data_input = [
        Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level,
        Vehicle_driver_relation, Driving_experience, Type_of_vehicle, Owner_of_vehicle,
        Service_year_of_vehicle, Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians,
        Road_alignment, Types_of_Junction, Road_surface_type, Road_surface_conditions,
        Light_conditions, Weather_conditions, Type_of_collision, Number_of_vehicles_involved,
        Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty,
        Age_band_of_casualty, Casualty_severity, Work_of_casualty, Fitness_of_casualty,
        Pedestrian_movement, Cause_of_accident, Hour_of_Day
    ]

    predicted_class = predict(data_input)

    if predicted_class[0] == 2:
        st.write("Predicted Injury : Slight Injury")
    elif predicted_class[0] == 1:
        st.write("Predicted Injury : Serious Injury")
    else:
        st.write("Predicted Injury : Fatal Injury")
