import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("ngos.csv")

# -----------------------------
# ML Model (Text Similarity)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
text_vectors = vectorizer.fit_transform(df['description'])

# Combine lat/lon + text vectors
features = np.hstack((
    text_vectors.toarray(),
    df[['lat', 'lon']].values
))

model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(features)

# -----------------------------
# UI
# -----------------------------
st.title("🌍 NGO Recommender App")

st.write("Find NGOs near you and support causes you care about ❤️")

# User Input
user_lat = st.number_input("Enter Latitude", value=28.6139)
user_lon = st.number_input("Enter Longitude", value=77.2090)
user_interest = st.text_input("What cause do you care about? (education, food, animals...)")

# -----------------------------
# Recommendation Logic
# -----------------------------
if st.button("Find NGOs"):

    user_text_vec = vectorizer.transform([user_interest]).toarray()
    user_features = np.hstack((user_text_vec, [[user_lat, user_lon]]))

    distances, indices = model.kneighbors(user_features)

    st.subheader("Recommended NGOs Near You:")

    for i in indices[0]:
        ngo = df.iloc[i]

        with st.container():
            st.markdown(f"### {ngo['name']}")
            st.write(f"📍 {ngo['city']}")
            st.write(f"💡 {ngo['category']}")
            st.write(f"📝 {ngo['description']}")
            st.write(f"🔗 {ngo['website']}")

            if st.button(f"View Details - {ngo['name']}"):
                st.session_state['selected'] = ngo['name']

# -----------------------------
# Detail Page
# -----------------------------
if 'selected' in st.session_state:
    selected_ngo = df[df['name'] == st.session_state['selected']].iloc[0]

    st.header(f"📖 {selected_ngo['name']} Details")

    st.write(f"📍 Location: {selected_ngo['city']}")
    st.write(f"🏷 Category: {selected_ngo['category']}")
    st.write(f"📝 Description: {selected_ngo['description']}")
    st.write(f"🌐 Website: {selected_ngo['website']}")

    st.write("📸 (Add images/videos here later)")
