
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title('Music Recommendation System')

# --- Data Loading (Simulate from your previous steps) ---
# In a real deployment, you might load this from a database or a pre-processed file
@st.cache_data # Cache the data loading to improve performance
def load_data():
    # Assuming the same path as in your notebook
    df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Assignments/Recommendation system/assignment_music_data_1.csv")
    user_artist_df = df.pivot_table(index='user', columns='artist', values='plays').fillna(0)
    return df, user_artist_df

df, user_artist_df = load_data()

# --- Recommendation Logic (from your notebook) ---
def get_collaborative_recommendations(target_user_id, user_artist_matrix, num_similar_users=3, num_recommendations=5):
    # Select the profile for the target user
    target_user_profile = user_artist_matrix.loc[target_user_id].values.reshape(1, -1)

    # Calculate cosine similarity between target user and all other users
    similarities = cosine_similarity(target_user_profile, user_artist_matrix)
    similarity_series = pd.Series(similarities[0], index=user_artist_matrix.index)

    # Remove the target user itself from the similarity list
    similarity_series = similarity_series.drop(target_user_id, errors='ignore')

    # Get the top N most similar users
    top_similar_users = similarity_series.nlargest(num_similar_users)

    # Get artists played by the target user
    target_user_played_artists = user_artist_matrix.loc[target_user_id][user_artist_matrix.loc[target_user_id] > 0].index

    # Get the profiles of these similar users
    similar_users_profiles = user_artist_matrix.loc[top_similar_users.index]

    # Aggregate plays for artists from similar users
    similar_users_artist_plays = similar_users_profiles.sum()

    # Remove artists already played by the target user
    recommendations = similar_users_artist_plays.drop(labels=target_user_played_artists, errors='ignore')

    # Get the top N recommendations
    final_recommendations = recommendations.nlargest(num_recommendations)
    return final_recommendations

def get_popularity_recommendations(df, num_recommendations=5):
    top_artists = df.groupby('artist')['plays'].sum().nlargest(num_recommendations)
    return top_artists

# --- Streamlit UI ---
st.sidebar.header('Choose a Recommendation Type')
recommendation_type = st.sidebar.radio(
    "Select a type of recommendation:",
    ('Popularity-Based', 'Collaborative Filtering')
)

if recommendation_type == 'Popularity-Based':
    st.subheader('Top 5 Most Popular Artists (For New Users)')
    popular_recs = get_popularity_recommendations(df, 5)
    st.write(popular_recs)
    st.markdown("""
    *Ideal for new users with no listening history.*
    """)
elif recommendation_type == 'Collaborative Filtering':
    st.subheader('Personalized Recommendations based on Similar Users')

    all_users = user_artist_df.index.tolist()
    selected_user = st.selectbox(
        'Select a User ID:',
        all_users
    )

    if selected_user:
        st.write(f"Recommendations for {selected_user}:")
        personalized_recs = get_collaborative_recommendations(selected_user, user_artist_df)
        st.write(personalized_recs)
        st.markdown("""
        *Based on what users with similar tastes have listened to.*
        """)

st.markdown("""
### How it works:
- **Popularity-Based**: Recommends the artists with the highest total plays across all users. Useful for cold-start scenarios.
- **Collaborative Filtering**: Finds users with similar listening patterns to the selected user and suggests artists that those similar users enjoy but the selected user hasn't heard yet.
""")
