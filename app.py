import os

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommendation System", layout="wide")
st.title("Music Recommendation System")

DATA_FILE = "assignment_music_data_1.csv"


# --- Data loading -----------------------------------------------------------
@st.cache_data
def load_data(source):
    """Read the ratings file and build a user x artist play-count matrix."""
    df = pd.read_csv(source)

    required = {"user", "artist", "plays"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    df = df.dropna(subset=["user", "artist"])
    df["plays"] = pd.to_numeric(df["plays"], errors="coerce").fillna(0)

    user_artist_df = df.pivot_table(
        index="user", columns="artist", values="plays", aggfunc="sum", fill_value=0
    )
    return df, user_artist_df


def get_data():
    """Load from the repo if the CSV is committed, otherwise ask for an upload."""
    if os.path.exists(DATA_FILE):
        return load_data(DATA_FILE)

    st.info(
        f"`{DATA_FILE}` isn't in the app directory. Upload it below, "
        "or commit it to the repository to load it automatically."
    )
    uploaded = st.file_uploader("Upload the listening-history CSV", type="csv")
    if uploaded is None:
        st.stop()
    return load_data(uploaded)


try:
    df, user_artist_df = get_data()
except ValueError as err:
    st.error(str(err))
    st.stop()


# --- Recommendation logic ---------------------------------------------------
def get_popularity_recommendations(df, num_recommendations=5):
    return df.groupby("artist")["plays"].sum().nlargest(num_recommendations)


def get_collaborative_recommendations(
    target_user_id, user_artist_matrix, num_similar_users=3, num_recommendations=5
):
    target_profile = user_artist_matrix.loc[target_user_id].values.reshape(1, -1)

    similarities = cosine_similarity(target_profile, user_artist_matrix)[0]
    similarity_series = pd.Series(similarities, index=user_artist_matrix.index)
    similarity_series = similarity_series.drop(target_user_id, errors="ignore")

    # Ignore users with no overlap at all.
    similarity_series = similarity_series[similarity_series > 0]
    if similarity_series.empty:
        return pd.Series(dtype=float)

    top_similar = similarity_series.nlargest(num_similar_users)

    # Weight each neighbour's play counts by how similar they are, so a heavy
    # listener with a loosely matching profile doesn't drown out a close match.
    neighbour_profiles = user_artist_matrix.loc[top_similar.index]
    weighted_scores = neighbour_profiles.mul(top_similar, axis=0).sum()

    target_row = user_artist_matrix.loc[target_user_id]
    already_played = target_row[target_row > 0].index
    recommendations = weighted_scores.drop(labels=already_played, errors="ignore")
    recommendations = recommendations[recommendations > 0]

    return recommendations.nlargest(num_recommendations)


# --- UI ---------------------------------------------------------------------
st.sidebar.header("Choose a Recommendation Type")
recommendation_type = st.sidebar.radio(
    "Select a type of recommendation:",
    ("Popularity-Based", "Collaborative Filtering"),
)
num_recommendations = st.sidebar.slider("Number of recommendations", 3, 20, 5)

if recommendation_type == "Popularity-Based":
    st.subheader(f"Top {num_recommendations} Most Popular Artists")
    popular_recs = get_popularity_recommendations(df, num_recommendations)
    st.bar_chart(popular_recs)
    st.dataframe(popular_recs.rename("total plays"))
    st.caption("Ideal for new users with no listening history.")

else:
    st.subheader("Personalized Recommendations Based on Similar Users")
    selected_user = st.selectbox("Select a User ID:", user_artist_df.index.tolist())

    if selected_user:
        recs = get_collaborative_recommendations(
            selected_user, user_artist_df, num_recommendations=num_recommendations
        )
        if recs.empty:
            st.warning(
                "No recommendations available for this user — they have either no "
                "listening history or no overlap with anyone else."
            )
        else:
            st.write(f"Recommendations for **{selected_user}**:")
            st.dataframe(recs.rename("recommendation score"))
        st.caption("Based on what users with similar tastes have listened to.")

st.markdown(
    """
### How it works
- **Popularity-Based** — ranks artists by total plays across all users. Useful for cold starts.
- **Collaborative Filtering** — finds the users whose listening patterns are closest to the
  selected user, then suggests artists those neighbours play that the selected user hasn't heard.
  Each neighbour's plays are weighted by their similarity score.
"""
)
