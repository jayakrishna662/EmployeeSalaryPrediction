from pyngrok import ngrok, conf
import os

# Set authtoken
conf.get_default().auth_token = " " #paste your ngrok authentication key

# Open a tunnel on port 8501
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url}")

# Launch  Streamlit app
os.system("streamlit run app.py")

Input Features:
The model takes the following attributes as input:

Age, Workclass, Final Weight (fnlwgt), Education Number, Marital Status, Occupation, Relationship, Race, Gender, Capital Gain, Capital Loss ,Hours per Week ,Native Country