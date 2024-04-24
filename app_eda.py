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

import librosa

import whisper
from whisper import load_model

# import pickle


# load and cache the data
@st.cache_data
def load_and_cache_data():
    # Load the pre-trained model
    data = pd.read_csv('./pickle/testing_set.csv', sep='|')

    return data

# Load it
data = load_and_cache_data()


# Main function
def main():
    st.write(f"Hello world!")

    # général
    st.write(f"Données utilisées : extraits audio + transcripts correspondants, (22? livres) ")
    # doughnut

    # texte
    # describe, hist

    # audio
    # origine
    # trimmed
    # mel spectro

# Run the app
main()



