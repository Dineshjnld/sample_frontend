import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Read the dataset
instadata = pd.read_csv("final-v1.csv")

# Drop the 'has_channel' column
instadata = instadata.drop(['has_channel'], axis=1)

# Split the data
x = instadata.drop(['is_fake'], axis=1)
y = instadata['is_fake']

# Train a Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(x, y)

# Create a StandardScaler
scaler = StandardScaler()

# Scale the data
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

def predict_fake_account(new_data_point):
    # Reorder columns to match the order during fit
    new_data_point = new_data_point[x.columns]
    
    # Scale the new data point using the same scaler
    new_data_point_scaled = pd.DataFrame(scaler.transform(new_data_point), columns=new_data_point.columns)
    predicted_class = classifier.predict(new_data_point_scaled)
    return predicted_class[0]

# Streamlit App
st.title("Fake Account Detection App")

# User Input Form
st.sidebar.header("User Input:")
username_length = st.sidebar.slider("Username Length", 0, 20, 5)
username_has_number = st.sidebar.selectbox("Username Has Number", [0, 1])
full_name_has_number = st.sidebar.selectbox("Full Name Has Number", [0, 1])
full_name_length = st.sidebar.slider("Full Name Length", 0, 50, 5)
is_private = st.sidebar.selectbox("Is Private", [0, 1])
is_joined_recently = st.sidebar.selectbox("Is Joined Recently", [0, 1])
is_business_account = st.sidebar.selectbox("Is Business Account", [0, 1])
has_guides = st.sidebar.selectbox("Has Guides", [0, 1])
has_external_url = st.sidebar.selectbox("Has External URL", [0, 1])
edge_follow = st.sidebar.slider("Edge Follow", 0, 100, 0)
edge_followed_by = st.sidebar.slider("Edge Followed By", 0, 100, 0)

# User Input Data
new_data_point = pd.DataFrame({
    'username_length': [username_length],
    'username_has_number': [username_has_number],
    'full_name_has_number': [full_name_has_number],
    'full_name_length': [full_name_length],
    'is_private': [is_private],
    'is_joined_recently': [is_joined_recently],
    'is_business_account': [is_business_account],
    'has_guides': [has_guides],
    'has_external_url': [has_external_url],
    'edge_follow': [edge_follow],
    'edge_followed_by': [edge_followed_by]
})

# Button to Trigger Prediction
if st.sidebar.button("Predict"):
    # Make Prediction
    predicted_class = predict_fake_account(new_data_point)

    # Display Prediction
    st.subheader("Prediction:")
    if predicted_class == 0:
        st.success("The account is likely to be original.")
    else:
        st.error("The account is likely to be fake.")
