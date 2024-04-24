# https://rosetta-stones.streamlit.app/


import subprocess

import streamlit as st
from audio_recorder_streamlit import audio_recorder
st.set_page_config(layout='wide')

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
import soundfile as sf

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


# load and cache the model
@st.cache_resource
def load_and_cache_model():
    # Load the pre-trained model
    model = load_model('tiny')

    return model

# Load the model
tiny_model = load_and_cache_model()


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
        labeldistance=1.3, pctdistance=0.85, colors=colors)

    plt.title(
    label=title,
    fontdict={"fontsize":17},
    pad=30
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

    page = 'info'

    with st.sidebar:
        st.header("Exploration")
        if st.button("Infos générales", key='info'):
            page = 'info'
        if st.button("Données texte", key='texte'):
            page = 'texte'
        if st.button("Données audio", key='audio'):
            page = 'audio'

    if page == 'info':
        st.write("Données disponibles : 2 426 extraits audio + les transcripts correspondants, provenant de 22 livres : ")
        fig1 = doughnut(data, 'title', 'Livres', width=10, height=10, nb_colors=data['title'].nunique())
        st.pyplot(fig=fig1, clear_figure=None, use_container_width=True)

    if page == 'texte':
        col1, col2 = st.columns(2)

        with col1: #describe
            # top margin ?
            st.write("Nombre de mots par extrait :")
            st.dataframe(test_data[['nb_words']].describe(),
                        column_config={"nb_words": st.column_config.NumberColumn(format="%d")})
        with col2: # hist
            fig2 = px.histogram(data_frame=test_data, x='nb_words', y=None, color=None,
                                title='Distribution')
            st.plotly_chart(fig2, use_container_width=True)

    if page == 'audio':
        # Selectionner aléatoirement
        number = random.randint(0, 2425)

        # Lire
        st.write(f" Voici un extrait aléatoire (numéro :blue[ {number}]) :")
        st.write(f"Transcript : {test_data['true_transcript'][number]}")

        # Ecouter
        exemple_path = test_data['path'][number]
        st.audio(exemple_path)

        # Visualiser
        exemple_audio_sf, sample_rate_sf = sf.read(exemple_path)

        # son d'origine
        fig1, ax = plt.subplots(figsize=(14, 4))
        librosa.display.waveshow(exemple_audio_sf, sr=sample_rate_sf)
        st.write("Son d'origine")
        st.pyplot(fig=fig1, clear_figure=None, use_container_width=True)

        # mel spectro
        mel = whisper.log_mel_spectrogram(audio).to(tiny_model.device)
        fig3 = plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel.numpy(), sr=sample_rate_sf, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        st.pyplot(fig=fig3, clear_figure=None, use_container_width=True)

# Run the app
main()



