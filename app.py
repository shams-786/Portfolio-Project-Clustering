from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

# Load the KMeans model and scaler
with open("./Model/Clustering Countries for Strategic Aid Allocation.pkl", 'rb') as file:
    kmeans = pickle.load(file)

scaler=StandardScaler()

# Load the country data
df = pd.read_csv('./Data/Country-data.csv')

# Define the features used for clustering
features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation',
            'life_expec', 'total_fer','import_export_ratio', 'gdpp_log',
            'regions_Africa','regions_Asia', 'regions_Caribbean', 'regions_Central America',
            'regions_Europe', 'regions_Middle East', 'regions_North America',
            'regions_Oceania', 'regions_South America']

# Prepare the data
df['import_export_ratio'] = df['imports'] / df['exports']
df['gdpp_log'] = np.log1p(df['gdpp'])

# One-hot encode the regions
regions = ['Africa','Asia', 'Caribbean', 'Central America',
            'Europe', 'Middle East', 'North America',
            'Oceania', 'South America', 'Other']
for region in regions:
    df[f'regions_{region}'] = 0


#Region Assignment

# Africa
df.loc[df['country'].isin(['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 
                   'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Cote d\'Ivoire', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Gabon', 'Gambia', 'Ghana', 
                   'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Morocco', 
                   'Mozambique', 'Mauritius', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Africa', 'Sudan', 
                   'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia']), 'regions_Africa'] = 1


# Asia
df.loc[df['country'].isin(['Afghanistan', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Georgia', 'India', 'Indonesia', 
                 'Japan', 'Kazakhstan','Kyrgyz Republic', 'Lao', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'Pakistan', 'Philippines', 
                 'Singapore', 'South Korea','Sri Lanka', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkmenistan', 'Uzbekistan', 'Vietnam']), 'regions_Asia'] = 1

# Caribbean
df.loc[df['country'].isin(['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Dominican Republic', 'Grenada', 'Haiti', 'Jamaica', 'St. Vincent and the Grenadines']), 'regions_Caribbean'] = 1

# Central America
df.loc[df['country'].isin(['Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Panama']), 'regions_Central America'] = 1

# Europe
df.loc[df['country'].isin(['Albania', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 
                   'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 
                   'Macedonia, FYR', 'Malta', 'Moldova', 'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'Serbia', 
                   'Slovak Republic', 'Slovenia', 'Spain', 'Sweden','Switzerland', 'Turkey', 'Ukraine', 'United Kingdom']), 'regions_Europe'] = 1

# Middle East
df.loc[df['country'].isin(['Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Qatar', 'Saudi Arabia', 'United Arab Emirates', 'Yemen']), 'regions_Middle East'] = 1

# North America
df.loc[df['country'].isin(['Canada', 'United States']), 'regions_North America'] = 1

# Oceania
df.loc[df['country'].isin(['Australia', 'Fiji', 'Kiribati', 'Micronesia, Fed. Sts.', 'New Zealand', 'Samoa', 'Solomon Islands', 'Tonga', 'Vanuatu']), 'regions_Oceania'] = 1

# South America
df.loc[df['country'].isin(['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']), 'regions_South America'] = 1

#Others
df['regions_Other'] = df['regions_Other'].fillna(1)  # Fill remaining NaN values with 1 for 'Other'

# Perform clustering on the loaded data
X = df[features]
X_scaled = scaler.fit_transform(X)
df['Cluster'] = kmeans.predict(X_scaled)

@app.route('/', methods=['GET'])
def home():
    return render_template('input_form.html', regions=regions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.form.to_dict()

        # Process the input data
        processed_data = process_input_data(data)

        # Extract features and convert to a NumPy array
        input_features = np.array([processed_data.get(feature, 0) for feature in features]).reshape(1, -1)

        # Preprocess input features
        input_features_scaled = scaler.transform(input_features)

        # Predict the cluster for the input data
        predicted_cluster = kmeans.predict(input_features_scaled)[0]

        # Find similar countries in the same cluster
        similar_countries = df[df['Cluster'] == predicted_cluster]

        # Sort similar countries by distance to the input data
        distances = np.linalg.norm(scaler.transform(similar_countries[features]) - input_features_scaled, axis=1)
        similar_countries = similar_countries.assign(Distance=distances)
        top_similar_countries = similar_countries.sort_values(by='Distance').head(5)['country'].tolist()

        # Determine the most important feature for the cluster
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_center = cluster_centers[predicted_cluster]
        feature_importances = np.abs(cluster_center - scaler.mean_)
        most_important_feature_index = np.argmax(feature_importances)
        most_important_feature = features[most_important_feature_index]

# Save the details of all similar countries to a JSON file
        json_filename = '../Data/similar_countries_details.json'
        features_with_distance = ['country'] + features + ['Distance']
        similar_countries_details = similar_countries[features_with_distance].to_dict(orient='records')
        with open(json_filename, 'w') as f:
            json.dump(similar_countries_details, f, indent=4)

        # Return the result
        return render_template('result.html', cluster=predicted_cluster, similar_countries=top_similar_countries,
                               important_feature=most_important_feature, json_filename=json_filename)


    except Exception as e:
        return jsonify({'error': str(e)})

def process_input_data(data):
    # Convert string inputs to appropriate types
    for feature in ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer']:
        if feature in data:
            data[feature] = float(data[feature])

    # Calculate derived features
    data['import_export_ratio'] = data['imports'] / data['exports'] if data['exports'] != 0 else 0
    data['gdpp_log'] = np.log1p(data['income'])  # Using income as GDP per capita

    # One-hot encode the region
    for region in regions:
        data[f'regions_{region}'] = 1 if data['region'] == region else 0

    return data

if __name__ == '__main__':
    app.run(debug=True)