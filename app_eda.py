# https://rosetta-stones.streamlit.app/


import subprocess

import streamlit as st
from audio_recorder_streamlit import audio_recorder

# paths, folders/files
import os, sys, random, re
from io import BytesIO

# math, dataframes
import numpy as np
import pandas as pd

# Visualisation
import matplotlib as matplot
import matplotlib.pyplot as plt
import plotly.express as px

# audio
import librosa
import librosa.display

# model
import whisper
from whisper import load_model

# import pickle


# load and cache the data
@st.cache_data
def load_and_cache_data():
    # Load the pre-trained model
    data = pd.read_csv('./pickle/info_generale.csv', sep='|')
    test_data = pd.read_csv('./pickle/test_data.csv', sep='|')

    return data, test_data

# Load it
data, test_data = load_and_cache_data()


def generate_random_pastel_colors(n):
    """
    Makes pretty colors... Sometimes :)
    Generates a list of n random pastel colors, represented as RGBA tuples.

    Parameters:
    n (int): The number of pastel colors to generate.

    Returns:
    list: A list of RGBA tuples representing random pastel colors.

    Example:
    >>> generate_random_pastel_colors(2)
    [(0.749, 0.827, 0.886, 1.0), (0.886, 0.749, 0.827, 1.0)]
    """
    colors = []
    for _ in range(n):
        # Generate random pastels
        red = round(random.randint(150, 250) / 255.0, 3)
        green = round(random.randint(150, 250) / 255.0, 3)
        blue = round(random.randint(150, 250) / 255.0, 3)

        # Create an RGB color tuple and add it to the list
        color = (red,green,blue, 1.0)
        colors.append(color)

    return colors


def doughnut(df, feature, title, width=6, height=6, nb_colors=2):
    """
    Affiche la répartition d'une feature sous forme de diagramme circulaire
    Display the distribution of a feature as a doughnut chart.
    Les couleurs sont aléatoires.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the feature.
        feature (str): The name of the feature to visualize.
        title (str): The title for the doughnut chart.
        width (int, optional): The width of the chart (default is 10).
        height (int, optional): The height of the chart (default is 10).

    The function creates a doughnut chart to visualize the distribution of the specified feature.
    If you don't like the colors, try running it again :)
    """
    colors = generate_random_pastel_colors(nb_colors)

    grouped_df = df.groupby(feature).size().to_frame("count_per_type").reset_index()
    pie = grouped_df.set_index(feature).copy()

    fig, ax = plt.subplots(figsize=(width, height))

    patches, texts, autotexts = plt.pie(x=pie['count_per_type'], autopct='%1.1f%%',
        startangle=-30, labels=pie.index, textprops={'fontsize':11, 'color':'#000'},
        labeldistance=1.25, pctdistance=0.85, colors=colors)

    plt.title(
    label=title,
    fontdict={"fontsize":17},
    pad=20
    )

    for text in texts:
        # text.set_fontweight('bold')
        text.set_horizontalalignment('center')

    # Customize percent labels
    for autotext in autotexts:
        autotext.set_horizontalalignment('center')
        autotext.set_fontstyle('italic')
        autotext.set_fontsize('10')

    #draw circle
    centre_circle = plt.Circle((0,0),0.7,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    return fig


# Main function
def main():

    page = 'app'

    with st.sidebar:
        st.header("Exploration")
        if st.button("Infos générales"):
            page = 'info'
        if st.button("Données texte"):
            page = 'texte'
        if st.button("Données audio"):
            page = 'audio'

        st.header("App")
        if st.button("Traduction"):
            page = 'app'

    if page == 'info':
        st.write("Données disponibles : 2 426 extraits audio + les transcripts correspondants, provenant de 22 livres : ")
        fig1 = doughnut(data, 'title', 'Livres', width=10, height=10, nb_colors=data['title'].nunique())
        st.pyplot(fig=fig1, clear_figure=None, use_container_width=True)

    if page == 'texte':
        col1, col2 = st.columns(2)

        with col1: #describe
            st.write("Nombre de mots par extrait :")
            st.dataframe(test_data[['nb_words']].describe(),
                        column_config={"nb_words": st.column_config.NumberColumn(format="%d")})
        with col2: # hist
            fig2 = px.histogram(data_frame=test_data, x='nb_words', y=None, color=None,
                                title='Distribution')
            st.plotly_chart(fig2, use_container_width=True)

    if page == 'audio':
        number = st.number_input("Choisissez le numéro d'un extrait : ",
                                 min_value = 0, max_value = 2425,
                                 value=None, placeholder="0 - 2425")
        # origine
        # trimmed
        # mel spectro

    if page == 'app':
        st.write("app")

# Run the app
main()



