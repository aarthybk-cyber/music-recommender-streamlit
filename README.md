# Music Recommendation System

## Project Overview

This project implements a basic music recommendation system using collaborative filtering and popularity-based approaches. It leverages user listening history to suggest artists that users might enjoy. The system is built in Python and can be deployed as an interactive web application using Streamlit.

## Features

*   **Data Loading and Preprocessing**: Loads music listening data, handles missing values, and pivots data into a user-artist interaction matrix.
*   **Popularity-Based Recommendations**: Provides recommendations based on the overall most played artists, ideal for new users or cold-start scenarios.
*   **Collaborative Filtering (User-Based)**: Generates personalized recommendations by finding users with similar listening patterns and suggesting artists that those similar users have enjoyed but the target user has not yet heard.
*   **Interactive Streamlit Application**: A web interface built with Streamlit allows users to select a recommendation type and a user ID to get real-time music suggestions.

## Technologies Used

*   **Python**: Programming language
*   **Pandas**: Data manipulation and analysis
*   **Scikit-learn**: For `cosine_similarity` in collaborative filtering
*   **Streamlit**: For creating the interactive web application
*   **Ngrok (optional for local testing)**: To expose the local Streamlit app to a public URL

## Getting Started

### 1. Data

The project uses a dataset named `assignment_music_data_1.csv`. This file should contain at least three columns: `user`, `artist`, and `plays`.

### 2. Local Setup (in Colab or local environment)

To run the project locally or in a Colab environment:

1.  **Clone the repository** (if applicable) or ensure `app.py` and `requirements.txt` are in your working directory.
2.  **Install dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    streamlit
    pandas
    scikit-learn
    ```
3.  **Run the Streamlit app**: 
    ```bash
    streamlit run app.py
    ```
    This will typically open the app in your browser at `http://localhost:8501`. If running in Colab, you might need a tool like `ngrok` to create a public URL (as demonstrated in the notebook).

## Deployment to Streamlit Community Cloud

The recommended way to deploy this application for public access is using Streamlit Community Cloud.

1.  **Create a GitHub Repository**: Place your `app.py` and `requirements.txt` files in the root of a new public or private GitHub repository (e.g., `music-recommender`).
2.  **Go to Streamlit Community Cloud**: Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3.  **Deploy a New App**: Click on "New app" and connect it to your GitHub repository.
    *   Select your repository.
    *   Ensure the branch is correctly set (e.g., `main`).
    *   The `Main file path` should be `app.py`.
4.  **Deploy**: Click "Deploy!". Streamlit Cloud will handle the installation of dependencies and launch your application, providing you with a public URL.

## Project Structure

*   `app.py`: The main Streamlit application script containing the recommendation logic and UI.
*   `requirements.txt`: Lists all Python dependencies required by the project.
*   `assignment_music_data_1.csv`: The dataset used for training and recommendations.
