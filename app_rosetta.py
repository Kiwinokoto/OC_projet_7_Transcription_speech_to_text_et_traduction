# https://image-classification-dog-breeds.streamlit.app/


import subprocess

# ui
try:
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder
except:
    print('Installing streamlit...')
    subprocess.run(["pip", "install", "streamlit"])
    subprocess.run(["pip", "install", "audio-recorder-streamlit"])
    print('Done')
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder

# paths, folders/files
import os, sys, random, re
from io import BytesIO

# math, dataframes
import numpy as np
import pandas as pd

try:
    import librosa
except:
    print('Installing librosa...')
    subprocess.run(["pip", "install", "librosa"])
    print('done')
    import librosa


try:
    import whisper
    from whisper import load_model
except:
    print('Installing whisper...')
    # Define the package URL
    package_url = "git+https://github.com/openai/whisper.git"

    # Use subprocess to run pip install command
    subprocess.run(["pip", "install", package_url])
    print('done')
    import whisper
    from whisper import load_model

# import pickle



# load and cache the model
@st.cache_resource
def load_and_cache_model():
    # Load the pre-trained model
    model = load_model('medium')

    return model

# Load the model
model = load_and_cache_model()


# Function to handle file upload
def handle_uploaded_file(model, recorded_speech):
    audio, sample_rate = librosa.load(BytesIO(recorded_speech))
    transcription = model.transcribe(audio, fp16 = False)['text']
    st.write(f"`{transcription}`")



# Main function
def main(model):

    # mic
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="6x",
        pause_threshold=5.0
    )

    # Handle file upload
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        st.write(f"`{type(audio_bytes)}`")
        st.write(f"`{len(audio_bytes)}`")

        handle_uploaded_file(model=model, recorded_speech=audio_bytes)

# Run the app
main(model=model)



