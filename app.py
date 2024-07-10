import streamlit as st 
import pandas as pd
import numpy as np
import joblib

# import liberaries for preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# preprocessing
class LabelTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda col: col.map(lambda x: 1 if x else 0))

class RoomSizeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.assign(Room_Size=X['squareMeters'] / X['numberOfRooms'])

class OwnerMappingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        owner_mapping = {
            1: 'New_1',
            2: 'New_2',
            3: 'New_3',
            4: 'New_4',
            5: 'New_5',
            6: 'New_6',
            7: 'New_7',
            8: 'New_8',
            9: 'New_9',
            10: 'New_10'
        }
        return X.assign(Prev_Cat=X['numPrevOwners'].map(owner_mapping))

class CityCodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['cityCode'] = X['cityCode'].astype(str).str.zfill(5)
        X['zone'] = X['cityCode'].str[0].astype(int)
        X['sub_zone'] = X['cityCode'].str[1].astype(int)
        X['sorting_district'] = X['cityCode'].str[:3].astype(int)
        X['post_office'] = X['cityCode'].str[3:].astype(int)
        X['cityCode'] = X['cityCode'].astype(int)
        return X

class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.assign(Age=X['made'].max() - X['made'])

num_features = ['squareMeters', 'numberOfRooms', 'made']
cat_features = ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom']


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

owner_pipeline = Pipeline([
    ('owner_mapping', OwnerMappingTransformer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
    ('label', LabelTransformer(), ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom']),
    ('room_size', RoomSizeTransformer(), ['squareMeters', 'numberOfRooms']),
    ('owner', owner_pipeline, ['numPrevOwners']),
    ('city_code', CityCodeTransformer(), ['cityCode']),
    ('age', AgeTransformer(), ['made'])
])


# Load the model
model = joblib.load('rand_reg.pkl')


st.title('Paris House Price Prediction')
st.write('This is a simple web app to predict the price of a house in Paris')

st.write('Please enter the details of the house:')

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    Area = st.number_input("Area (Square Meters)", max_value=6071330, min_value=85, value=85)
    Rooms = st.number_input("Number of Rooms", max_value=100, min_value=1, value=1)
    Yard = st.selectbox("Has Yard?", ["Yes", "No"])
    Pool = st.selectbox("Has Pool?", ["Yes", "No"])
    Floors = st.number_input("Number of floors", max_value=6000, min_value=1, value=1)
    CityCode = st.text_input("City Code")
    CityPartRange = st.select_slider("City Part Range", options=[1,2,3,4,5,6,7,8,9,10])
    GuestRoom = st.selectbox("Has Guest Room?", ["Yes", "No"])
    
with col2:
    PreviousOwners = st.select_slider("Number of Previous Owners", options=[0,1,2,3,4,5,6,7,8,9,10])
    YearOfConstruction = st.number_input("Year of Construction", max_value=2021, min_value=1990, value=1990)
    NewlyBuilt = st.selectbox("Is Newly Built?", ["Yes", "No"])
    StormProtector = st.selectbox("Has Storm Protector?", ["Yes", "No"])
    BasementArea = st.number_input("Basement Area (Square Meters)", max_value=91992, min_value=4, value=4)
    Attic = st.number_input("Attic (Square Meters)", max_value=96381, min_value=1, value=1)
    GarageSize = st.number_input("Garage Size (Square Meters)", max_value=9017, min_value=4, value=4)
    StorageRoom = st.selectbox("Has Storage Room?", ["Yes", "No"])
    

user_input = pd.DataFrame({
    'squareMeters': [Area],
    'numberOfRooms': [Rooms],
    'hasYard': [1 if Yard == 'Yes' else 0],
    'hasPool': [1 if Pool == 'yes' else 0],
    'floor' : [Floors],
    'cityCode': [CityCode],
    'cityPartRange': [CityPartRange],
    'numPrevOwners': [PreviousOwners],
    'made': [YearOfConstruction],
    'isNewBuilt': [1 if NewlyBuilt == 'Yes' else 0], 
    'hasStormProtector': [1 if StormProtector == 'Yes' else 0],
    'basement': [BasementArea],
    'attic': [Attic],
    'garage': [GarageSize],
    'hasStorageRoom': [1 if StorageRoom == 'Yes' else 0],
    'hasGuestRoom': [1 if GuestRoom == 'Yes' else 0]
})

if st.button('Predict'):    
    prediction = model.predict(user_input)
    st.success(f'The predicted price is: {prediction[0]:,.2f} EUR')