# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# import pickle

# # Load the dataset
# df = pd.read_csv('laptop_data.csv')

# # Data Preprocessing
# # Remove the 'Unnamed: 0' column as it's just an index
# df = df.drop(columns=['Unnamed: 0'])

# # Convert 'Ram' and 'Weight' to numeric after stripping the non-numeric part
# df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
# df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# # Handle the 'Memory' column (which has mixed storage types like 'SSD', 'HDD')
# df['Memory'] = df['Memory'].apply(lambda x: ' '.join(sorted(x.split(' '))))

# # Split the data into features and target
# X = df.drop(columns=['Price','Cpu'])
# y = df['Price']

# # Preprocess categorical columns
# categorical_cols = ['Company', 'TypeName', 'ScreenResolution', 'Memory', 'Gpu', 'OpSys']
# numerical_cols = ['Inches', 'Ram', 'Weight']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_cols),
#         ('cat', OneHotEncoder(), categorical_cols)
#     ])

# # Simple Linear Regression Model
# lin_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                    ('model', LinearRegression())])

# # Random Forest Model
# rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

# # Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fit the models
# lin_reg_pipeline.fit(X_train, y_train)
# rf_pipeline.fit(X_train, y_train)

# # Save the preprocessor and the models
# with open('df.pkl', 'wb') as f:
#     pickle.dump(preprocessor, f)

# with open('pipe.pkl', 'wb') as f:
#     pickle.dump(rf_pipeline, f)

# print("Models trained and saved successfully.")










import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('laptop_data.csv',encoding='latin1')

# Data Preprocessing
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])


df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
df['Memory'] = df['Memory'].apply(lambda x: ' '.join(sorted(x.split(' '))))

# Convert categorical columns to numerical using LabelEncoder
categorical_cols = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the label encoder for inverse transform if needed

# Split the data into features and target
X = df.drop(columns=['Price'])
y = df['Price']

# Scale numerical features
scaler = StandardScaler()
X[['Inches', 'Ram', 'Weight']] = scaler.fit_transform(X[['Inches', 'Ram', 'Weight']])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model, scaler, and label encoders
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }, f)

print("Model trained and saved successfully.")


import pandas as pd
import pickle

# Load the saved model, scaler, and label encoders
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    label_encoders = data['label_encoders']

def encode_label(label, encoder):
    """Encodes a label with a fallback for unseen labels."""
    try:
        return encoder.transform([label])[0]
    except ValueError:
        # Fallback: Assign an existing label with similar characteristics or the most frequent label
        print(f"Warning: Unseen label '{label}' for column. Assigning fallback value.")
        return encoder.transform([encoder.classes_[0]])[0]  # Assigning the first known class

# Sample data for prediction with unseen labels handling
sample_data = pd.DataFrame({
    'Company': [encode_label('Dell', label_encoders['Company'])],
    'TypeName': [encode_label('Gaming', label_encoders['TypeName'])],
    'ScreenResolution': [encode_label('1920x1080', label_encoders['ScreenResolution'])],
    'Cpu': [encode_label('Intel Core i7', label_encoders['Cpu'])],
    'Memory': [encode_label('8GB SSD', label_encoders['Memory'])],
    'Gpu': [encode_label('NVIDIA GTX 1650', label_encoders['Gpu'])],
    'OpSys': [encode_label('Windows 10', label_encoders['OpSys'])],
    'Inches': [15.6],
    'Ram': [8],
    'Weight': [2.3]
})

# Ensure the columns are in the same order as during training
feature_order = ['Company', 'TypeName','Inches' ,'ScreenResolution', 'Cpu','Ram', 'Memory', 'Gpu', 'OpSys',  'Weight']
sample_data = sample_data[feature_order]

# Scale the numerical features
sample_data[['Inches', 'Ram', 'Weight']] = scaler.transform(sample_data[['Inches', 'Ram', 'Weight']])

# Predict the price using the model
predicted_price = model.predict(sample_data)
print("Predicted Price for the sample data:", predicted_price[0])
 




















































# from flask import Flask, render_template, request
# import pickle
# import pandas as pd
# import numpy as np

# # Load the preprocessor and model pipeline
# preprocessor = pickle.load(open("df.pkl", 'rb'))
# pipe = pickle.load(open('pipe.pkl', 'rb'))

# app = Flask(__name__)

# @app.route("/")
# def index():
#     # Prepare the dropdown options
#     data_dict = {
#         'company': ['Apple', 'Dell', 'HP', 'Lenovo', 'Acer', 'Asus'],  # Example values
#         'typename': ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible'],
#         'ram': [4, 8, 16, 32],
#         'weight': [1.0, 4.0],
#         'os': ['Windows', 'macOS', 'Linux', 'No OS'],
#         'touchscreen': [0, 1],
#         'isips': [0, 1],
#         'cpubrand': ['Intel Core i7', 'AMD'],
#         'ssd': [128, 256, 512, 1024],
#         'hdd': [0, 500, 1000],
#         'Gpu_brand': ['Intel', 'Nvidia', 'AMD'],
#         'resolution': ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
#         'prediction': False
#     }
#     return render_template('index.html', data_dict=data_dict)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input data from form
#     brand = request.form['brand']
#     typename = request.form['typename']
#     ram = int(request.form['ram'])
#     weight = float(request.form['weight'])
#     touchscreen = int(request.form['touchscreen'])
#     ips = int(request.form['IPS'])
#     screen_size = float(request.form['screen_size'])
#     resolution = request.form['resolution']
#     cpu = request.form['cpu']
#     ssd = int(request.form['ssd'])
#     hdd = int(request.form['hdd'])
#     gpu = request.form['gpu']
#     os = request.form['os']

#     # Process resolution to get PPI
#     X_resolution = int(resolution.split('x')[0])
#     y_resolution = int(resolution.split('x')[1])
#     ppi = ((X_resolution**2) + (y_resolution**2))**0.5 / screen_size

#     # Construct Memory column
#     memory = f"{ssd}GB SSD {hdd}GB HDD"



# # sample_data = pd.DataFrame({
# #     'Company': ['Dell'],
# #     'TypeName': ['Gaming'],
# #     'ScreenResolution': ['1920x1080'],
# #     'Cpu': ['Intel Core i7'],
# #     'Memory': ['8GB SSD'],
# #     'Gpu': ['NVIDIA GTX 1650'],
# #     'OpSys': ['Windows 10'],
# #     'Inches': [15.6],
# #     'Ram': [8],
# #     'Weight': [2.3]
# # })




#     # Create a DataFrame for the input features
#     input_data = pd.DataFrame({
#         'Company': [brand],
#         'TypeName': [typename],
#         'Inches': [screen_size],
#         'Ram': [ram],
#         'Weight': [weight],
#         'ScreenResolution': [resolution],  # Adding the original resolution
#         'Touchscreen': [touchscreen],
#         'IPS': [ips],
#         'PPI': [ppi],
#         'Memory': [memory],  # Constructed Memory
#         # 'Cpu': [cpu],
#         'Gpu': [gpu],
#         'OpSys': [os]
#     })

#     # Preprocess the input data and make a prediction
#     transformed_input = preprocessor.transform(input_data)
#     prediction = pipe.predict(transformed_input)[0]

#     # Prepare the data_dict for rendering
#     data_dict = {
#         'company': ['Apple', 'Dell', 'HP', 'Lenovo', 'Acer', 'Asus'],
#         'typename': ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible'],
#         'ram': [4, 8, 16, 32],
#         'weight': [1.0, 4.0],
#         'os': ['Windows', 'macOS', 'Linux', 'No OS'],
#         'touchscreen': [0, 1],
#         'isips': [0, 1],
#         # 'cpubrand': ['Intel Core i7', 'AMD'],
#         'ssd': [128, 256, 512, 1024],
#         'hdd': [0, 500, 1000],
#         'Gpu_brand': ['Intel', 'Nvidia', 'AMD'],
#         'resolution': ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
#         'prediction': prediction
#     }

#     return render_template("index.html", data_dict=data_dict)

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# if __name__ == '__main__':
#     app.run(debug=True)
