import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
Dataset is from: [Palmer Penguins Library](https://github.com/dataprofessor/data/blob/master/penguins_cleaned.csv)
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

#for i in range(4):
#    print(min(X.iloc[:,i]),max(X.iloc[:,i]))
#
#ANS: 
#    32.1 59.6
#    13.1 21.5
#    172 231
#    2700 6300

def user_input_features():
    island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = user_input_features()

# Convert categorical variables to dummy variables
sex_female = []
sex_male = []
island_Biscoe = []
island_Dream = []
island_Torgersen = []
sex = ['sex_female','sex_male']
island = ['island_Biscoe','island_Dream','island_Torgersen']

temp = input_df[['sex','island']]
for i in temp.values:
    if i[0] == 'female':
        sex_female.append(1)
        sex_male.append(0)
    else:
        sex_female.append(0)
        sex_male.append(1)
    
    if i[1] == 'Biscoe':
        island_Biscoe.append(1)
        island_Dream.append(0)
        island_Torgersen.append(0)
    elif i[1] == 'Dream':
        island_Biscoe.append(0)
        island_Dream.append(1)
        island_Torgersen.append(0)
    else:
        island_Biscoe.append(0)
        island_Dream.append(0)
        island_Torgersen.append(1)

# Merge dummy variables with other features
temp_data = {'sex_female':sex_female,
             'sex_male':sex_male,
             'island_Biscoe':island_Biscoe,
             'island_Dream':island_Dream,
             'island_Torgersen':island_Torgersen
            }
temp_data = pd.DataFrame(temp_data)
test_data = input_df.drop(['sex','island'],axis=1).reset_index(drop=True)
input_df = pd.concat([test_data,temp_data],axis=1)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Selected features from the sidebar:')
    st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
prediction = pd.DataFrame(penguins_species[prediction],columns=['Predicted Class'])
st.write(prediction)

st.subheader('Prediction Probability')
prediction_proba = pd.DataFrame(prediction_proba,columns=['Adelie','Chinstrap','Gentoo'])
st.write(prediction_proba)
