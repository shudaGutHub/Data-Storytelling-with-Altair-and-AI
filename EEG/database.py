import sqlite3
import pandas as pd
path_db = "https://github.com/alod83/Data-Storytelling-with-Python-Altair-and-Generative-AI/blob/main/patient_database.db"

# Reconnect to the database to fetch both patients and visits data
conn = sqlite3.connect(path_db)

# Fetch patient data
patients_query = "SELECT * FROM patients"
patients_data = pd.read_sql_query(patients_query, conn)

# Fetch visit data (assuming there is a visits table)
visits_query = "SELECT * FROM visits"
visits_data = pd.read_sql_query(visits_query, conn)

# Merge the patients and visits data based on the foreign key subject_elm_id
merged_data = pd.merge(
    patients_data, 
    visits_data, 
    left_on='subject_elm_id', 
    right_on='subject_elm_id', 
    how='left'
)

# Create a dictionary where each patient is mapped to their visit dates
patients_visits_dict = merged_data.groupby('subject_elm_id')['visit_date'].apply(list).to_dict()

# Close the connection
conn.close()

# Display the dictionary to the user
for patient in patients_visits_dict.keys():
    print(patient)

