import html
import os

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "assignment_music_data_1.csv"

st.set_page_config(
    page_title="Listening Chart",
    page_icon="♫",
    layout="centered",
)


# --- Styling ----------------------------------------------------------------
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;800&display=swap');

.stApp {
    background: #F4F6F5;
    color: #16211E;
    font-family: 'Archivo', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background: #EDF1EF;
    border-right: 1px solid #D5DDD9;
}

[data-testid="stSidebar"] * { color: #16211E; }

.masthead { padding: 0.5rem 0 1.75rem; }

.masthead h1 {
    font-family: 'Archivo', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.02;
    letter-spacing: -0.035em;
    margin: 0;
    color: #16211E;
}

.masthead p {
    margin: 0.6rem 0 0;
    max-width: 54ch;
    font-size: 0.95rem;
    line-height: 1.55;
    color: #5F736C;
}

.section-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    border-bottom: 1px solid #D5DDD9;
    padding-bottom: 0.5rem;
    margin-bottom: 0.25rem;
}

.section-head h2 {
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    margin: 0;
    color: #16211E;
}

.section-head span { font-size: 0.8rem; color: #5F736C; }

ol.chart { list-style: none; margin: 0; padding: 0; }

ol.chart li {
    display: grid;
    grid-template-columns: 2.6rem 1fr auto;
    align-items: center;
    gap: 0 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid #E2E8E5;
}

.rank {
    font-weight: 800;
    font-size: 1.5rem;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    color: #A66A11;
    text-align: right;
}

.name {
    display: block;
    font-weight: 500;
    font-size: 1.02rem;
    letter-spacing: -0.01em;
    margin-bottom: 0.45rem;
    overflow-wrap: anywhere;
}

.track {
    display: block;
    height: 5px;
    background: #E2E8E5;
    border-radius: 3px;
    overflow: hidden;
}

.fill {
    display: block;
    height: 100%;
    background: #A66A11;
    border-radius: 3px;
    transform-origin: left center;
    animation: grow 420ms cubic-bezier(0.2, 0.7, 0.3, 1);
}

@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

@media (prefers-reduced-motion: reduce) {
    .fill { animation: none; }
}

.value {
    font-variant-numeric: tabular-nums;
    font-size: 0.9rem;
    color: #5F736C;
    white-space: nowrap;
}

.neighbours {
    margin: 1.5rem 0 0;
    padding: 0.9rem 1.1rem;
    background: #EBEFED;
    border-radius: 8px;
    font-size: 0.87rem;
    color: #5F736C;
    line-height: 1.7;
}

.neighbours b { color: #16211E; font-weight: 500; }

.footnote {
    margin-top: 2.5rem;
    padding-top: 1.1rem;
    border-top: 1px solid #D5DDD9;
    font-size: 0.85rem;
    line-height: 1.65;
    color: #5F736C;
    max-width: 62ch;
}

@media (max-width: 480px) {
    .masthead h1 { font-size: 2rem; }
    ol.chart li { grid-template-columns: 1.9rem 1fr auto; }
    .rank { font-size: 1.2rem; }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# --- Data loading -----------------------------------------------------------
@st.cache_data
def load_data(source):
    """Read the listening history and build a user x artist play-count matrix."""
    df = pd.read_csv(source)

    required = {"user", "artist", "plays"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"This file needs a column named {' and '.join(sorted(missing))}. "
            "Check the header row and upload it again."
        )

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

    st.markdown(
        '<div class="masthead"><h1>Listening Chart</h1>'
        "<p>Add a listening history to get started. The file needs one row per "
        "user and artist, with columns for user, artist and plays.</p></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Choose a CSV", type="csv", label_visibility="collapsed")
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
    """Return (recommendations, neighbours) for one user."""
    target_profile = user_artist_matrix.loc[target_user_id].values.reshape(1, -1)

    similarities = cosine_similarity(target_profile, user_artist_matrix)[0]
    similarity_series = pd.Series(similarities, index=user_artist_matrix.index)
    similarity_series = similarity_series.drop(target_user_id, errors="ignore")

    # Ignore users who share no artists at all with the target.
    similarity_series = similarity_series[similarity_series > 0]
    if similarity_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    neighbours = similarity_series.nlargest(num_similar_users)

    # Weight each neighbour's play counts by how similar they are, so a heavy
    # listener with a loosely matching profile can't drown out a close match.
    weighted_scores = user_artist_matrix.loc[neighbours.index].mul(neighbours, axis=0).sum()

    target_row = user_artist_matrix.loc[target_user_id]
    already_played = target_row[target_row > 0].index
    recommendations = weighted_scores.drop(labels=already_played, errors="ignore")
    recommendations = recommendations[recommendations > 0]

    return recommendations.nlargest(num_recommendations), neighbours


# --- Rendering --------------------------------------------------------------
def render_chart(series, decimals=0):
    """Draw a ranked list with a proportional bar behind each value."""
    top = series.max()
    rows = []
    for position, (name, value) in enumerate(series.items(), start=1):
        width = 0.0 if top <= 0 else value / top * 100
        rows.append(
            f'<li><span class="rank">{position}</span>'
            f'<span><span class="name">{html.escape(str(name))}</span>'
            f'<span class="track"><span class="fill" style="width:{width:.1f}%"></span></span></span>'
            f'<span class="value">{value:,.{decimals}f}</span></li>'
        )
    st.markdown(f'<ol class="chart">{"".join(rows)}</ol>', unsafe_allow_html=True)


def section_head(title, note):
    st.markdown(
        f'<div class="section-head"><h2>{html.escape(title)}</h2>'
        f"<span>{html.escape(note)}</span></div>",
        unsafe_allow_html=True,
    )


# --- Sidebar ----------------------------------------------------------------
st.sidebar.markdown("### Chart settings")
mode = st.sidebar.radio(
    "Chart",
    ("Most played", "Picked for a listener"),
    label_visibility="collapsed",
)
chart_length = st.sidebar.slider("Chart length", 3, 20, 5)

listener = None
neighbour_count = 3
if mode == "Picked for a listener":
    listener = st.sidebar.selectbox("Listener", user_artist_df.index.tolist())
    neighbour_count = st.sidebar.slider("Listeners to compare against", 1, 10, 3)

st.sidebar.markdown(
    '<p style="font-size:0.8rem;color:#5F736C;margin-top:1.5rem;line-height:1.6;">'
    f"{len(user_artist_df):,} listeners &middot; {user_artist_df.shape[1]:,} artists "
    f"&middot; {df['plays'].sum():,.0f} plays</p>",
    unsafe_allow_html=True,
)


# --- Main -------------------------------------------------------------------
if mode == "Most played":
    st.markdown(
        '<div class="masthead"><h1>Most played</h1>'
        "<p>The artists with the highest total play counts across every listener "
        "in the dataset. This is what a brand new listener sees, before there is "
        "any history to learn from.</p></div>",
        unsafe_allow_html=True,
    )

    section_head(f"Top {chart_length}", "total plays")
    render_chart(get_popularity_recommendations(df, chart_length))

else:
    st.markdown(
        f'<div class="masthead"><h1>Picked for {html.escape(str(listener))}</h1>'
        "<p>Artists played by the listeners whose taste most closely matches this "
        "one, filtered down to what this listener hasn\u2019t heard yet.</p></div>",
        unsafe_allow_html=True,
    )

    recommendations, neighbours = get_collaborative_recommendations(
        listener,
        user_artist_df,
        num_similar_users=neighbour_count,
        num_recommendations=chart_length,
    )

    if recommendations.empty:
        st.markdown(
            '<div class="neighbours">Nothing to suggest for this listener yet. '
            "They either have no plays recorded, or share no artists with anyone "
            "else in the dataset. Try another listener, or widen the comparison "
            "in the sidebar.</div>",
            unsafe_allow_html=True,
        )
    else:
        section_head(f"Top {len(recommendations)}", "match score")
        render_chart(recommendations, decimals=1)

        matches = " &middot; ".join(
            f"<b>{html.escape(str(name))}</b> {score:.0%}"
            for name, score in neighbours.items()
        )
        st.markdown(
            f'<div class="neighbours">Closest listening profiles: {matches}</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footnote">Most played ranks artists by total plays, which works '
    "for anyone the system knows nothing about. Picked for a listener compares "
    "listening profiles with cosine similarity, then scores each candidate artist "
    "by the neighbours\u2019 play counts weighted by how close a match each "
    "neighbour is.</div>",
    unsafe_allow_html=True,
)
