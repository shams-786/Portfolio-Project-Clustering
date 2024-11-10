# Portfolio-Project-Clustering
Clustering Countries for Strategic Aid Allocation
## Project Structure
- Portfolio-Project-Clustering/
  - Data/
    - Country-data.csv
    - similar_countries_details.json
  - Model/
    - Clustering Countries for Strategic Aid Allocation.pkl
  - Others
    - Folder Staructure.jpg
    - Input form.jpg
    - Input.jpg
    - Result.jpg
  - SourceFiles
    - Clustering_Countries_for_Strategic_Aid_Allocation.ipynb
    - Countries for Strategic Aid Allocation.twbx
  - Templates/
    - input_form.html
    - result.html
  - app.py
  - README.md
  - requirements.txt

## Setup

1. **Clone the repository:**
   git clone https://github.com/shams-786/Portfolio-Project-Clustering.git

2. **Create a virtual environment:**
   python3 -m venv .venv
   source .venv/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt

## Hosting the Models
1. **run the app.py script:**
   python app.py
   ```
   This script will start a Flask API server that loads the saved models and serves predictions and successful deployment shows the message "Enter Country Details" on the screen.
   ```
## Validating the Models
1. **Test the API:**
+ Send a POST request to http://127.0.0.1:5000/predict with JSON data containing the features required for prediction.

+ This response will contain the predicted Top 5 Countries with Similar Features based on the input features.

## Technical Blog
For a detailed analysis and insights from the project, see the Technical Blog at https://medium.com/@msshams786/clustering-countries-for-strategic-aid-allocation-using-unsupervised-machine-learning-47e7a7194be8

**Tableau Dashboard**
For the Tableau Dashboard it can be accessed/viewed at https://public.tableau.com/app/profile/mohd.shahid.shams8690/viz/CountriesforStrategicAidAllocation/StrategicAidAllocation

## Troubleshooting
**TBD**







