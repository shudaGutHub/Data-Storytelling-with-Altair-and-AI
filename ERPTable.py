import json
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from scipy.stats import norm
from IPython.display import HTML
import pathlib
import pandas as pd
import fasthtml as fh
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

#DATA_DIR = pathlib.Path(r'C:\Users\Saleem\ERPReportProject\data\AM001')
#DATA_DIR = pathlib.Path(r"C:\Users\salee\projects\ERPReportProject\data")

DATA_DIR = pdata = pathlib.Path(r"C:\Users\salee\projects\ERPReportProject\data\AM001")
fname = "erp-report.json"

file_erp = DATA_DIR/fname

# Check if the file exists
if not file_erp.exists():
    raise FileNotFoundError(f"File {file_erp} not found")

# Define Pydantic models
class Scores(BaseModel):
    score: Optional[float]
    zScore: Optional[float]

class Subject(BaseModel):
    condition: str
    ERPv_zScore: Optional[float]
    scores: Dict[str, Scores]
    capsule: Optional[Dict]
    erp: Optional[Dict]

class CognitiveFunctionDetails(BaseModel):
    groupSize: int
    gender: str
    ageRange: Dict[str, int]
    erpComponent: str
    subject: Subject
class Condition(BaseModel):
    erp: Optional[Dict]
    group: Optional[str]
    groupSize: Optional[int]

class CognitiveFunction(BaseModel):
    conditions: Dict[str, Condition]
    erpComponent: str
    group: Optional[str]
    groupSize: Optional[int]
    

        



    


class Results(BaseModel):
    cognitiveFunctions: Dict[str, CognitiveFunctionDetails]

class ERPReport(BaseModel):
    results: Results
    reportInfo: Optional[Dict    ]

    
# Cognitive function labels mapping
cognitive_function_labels = {
    "PS": "Latency/Speed",
    "NR": "Power/NeuralRecruitment",
    "P50": "P50",
    "N100": "N100",
    "P200": "P200",
    "P3a": "P3a",
    "P3b": "P3b",
    "N200": "N200",
    "MI": "Movement Inhibition",
    "FOI": "Filtering of Information",
    "EAP": "Early Auditory Processing",
    "ADI": "Auditory Discrimination",
    "AP": "Auditory Processing",
    "SA": "Sensory Attention",
    "ATML": "Attentional Modulation",
    "PAWM": "Prefrontal Activation and Working Memory",
    "RSAOB": "Response Selection and Auditory Oddball"
}

# Load the JSON data
with open(file_erp, 'r') as f:
    erp_data = json.load(f)



# Parse the JSON data into Pydantic models
erp_report = ERPReport(**erp_data)
records = []
for condition, details in erp_report.results.cognitiveFunctions.items():
    condition_name = details.subject.condition
    erp_component = details.erpComponent
    for cognitive_function_key, cognitive_function_details in details.subject.scores.items():
        cognitive_function_label = cognitive_function_labels.get(cognitive_function_key, cognitive_function_key)
        scores = cognitive_function_details.score
        z_score = cognitive_function_details.zScore

        percentile_score = norm.cdf(z_score) * 100 if z_score is not None else None  # Convert z-score to percentile

        neural_consistency_score = details.subject.ERPv_zScore   
        
        neural_consistency_percentile = norm.cdf(neural_consistency_score) * 100 if neural_consistency_score is not None else None
        
        
        records.append({
            'condition': condition_name,
            'eegMeasurement': cognitive_function_label,
            'erpComponent': erp_component,
            'scores': scores,
            'zScore': z_score,
            'percentileScore': percentile_score,
            'neuralConsistencyScore': neural_consistency_score,
            'neuralConsistencyPercentile':neural_consistency_percentile

        })

# Create the dataframe
df = pd.DataFrame(records)
df.set_index(['condition', 'erpComponent','eegMeasurement'], inplace=True)  

def create_sparkline(percentile):
    fig, ax = plt.subplots(figsize=(2, 0.5))

    # Generate data points for the Gaussian curve
    x = np.linspace(0, 100, 1000)
    y = norm.pdf(x, 50, 15)  # Assuming mean=50 and std_dev=15 for the Gaussian curve

    # Plot the Gaussian curve
    ax.plot(x, y, 'k-', lw=0.5)

    # Plot the percentile point
    ax.plot(percentile, norm.pdf(percentile, 50, 15), 'ro')

    # Add threshold indicator marks at 10 and 90
    ax.axvline(x=10, color='blue', linestyle='--', lw=0.5)
    ax.axvline(x=90, color='blue', linestyle='--', lw=0.5)

    # Set limits and remove axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(y))
    ax.axis('off')

    # Save the plot to a BytesIO object and return as base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# Add sparklines to dataframe
df['sparkline'] = df['percentileScore'].apply(create_sparkline)

# Convert sparklines to HTML image tags
df['sparkline'] = df['sparkline'].apply(lambda x: f'<img src="data:image/png;base64,{x}" />')

df_reset = df.reset_index()

df_pivot = df_reset.pivot(index =['erpComponent','condition'], columns ='eegMeasurement',values=['scores', 'zScore', 'percentileScore', 'neuralConsistencyScore', 'sparkline'])


# Flatten the multi-level columns
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

'''
Add gaussian sparklines to dataframe from distcurve.py
'''
def generate_gaussian_curve(percentile_score, mean=0, std_dev=1, distfunc=norm):
    fig, ax = plt.subplots(figsize=(2, 0.5))

    # Generate data points for the Gaussian curve
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
    y = distfunc.pdf(x, mean, std_dev)

    # Plot the Gaussian curve
    ax.plot(x, y, 'k-', lw=0.5)

    # Plot the percentile point
    percentile_value = distfunc.ppf(percentile_score / 100, mean, std_dev)
    ax.plot(percentile_value, distfunc.pdf(percentile_value, mean, std_dev), 'ro')

    # Add threshold indicator marks at 10 and 90
    ax.axvline(x=distfunc.ppf(0.1, mean, std_dev), color='blue', linestyle='--', lw=0.5)
    ax.axvline(x=distfunc.ppf(0.9, mean, std_dev), color='blue', linestyle='--', lw=0.5)

    # Set limits and remove axes
    ax.set_xlim(mean - 3*std_dev, mean + 3*std_dev)
    ax.set_ylim(0, max(y))
    ax.axis('off')

    # Save the plot to a BytesIO object and return as base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

df['gaussian_sparkline'] = df['percentileScore'].apply(generate_gaussian_curve)

# Example usage of the generate_gaussian_curve function
# df['gaussian_sparkline'] = df['percentileScore'].apply(generate_gaussian_curve)

def style_dataframe(df):
    styler = df.style.set_table_styles(
        [
            {'selector': 'thead th', 'props': [('background-color', '#f7f7f9'), ('color', '#333'), ('border', '1px solid #ddd')]},
            {'selector': 'tbody td', 'props': [('border', '1px solid #ddd')]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#f1f1f1')]},
        ]
    ).set_properties(**{
        'text-align': 'center',
        'padding': '10px'
    })
    return styler

# Function to render dataframe as HTML with footer
def render_dataframe_with_footer(df, footer_text):
    styled_df = style_dataframe(df)
    html_table = styled_df.to_html(escape=False)
    footer_html = f"<tfoot><tr><td colspan='{len(df.columns)}' style='text-align:center;'>{footer_text}</td></tr></tfoot>"
    html_table = html_table.replace('</tbody>', f'</tbody>{footer_html}')
    return html_table
from datetime import datetime
# Get the current time
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Render the DataFrame with the current time as a footer
render_dataframe_with_footer(df, f"Report generated on: {current_time}")
# Render the DataFrame with the current time as a footer
html_content = render_dataframe_with_footer(df, f"Report generated on: {current_time}")


# Save as HTML
output_html_file = pathlib.Path(r'C:\Users\salee\projects\ERPReportProject\output.html')
with open(output_html_file, 'w') as f:
    f.write(html_content)
