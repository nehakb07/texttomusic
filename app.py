import streamlit as st
from transformers import AutoProcessor, AutoModelForTextToWaveform, AutoTokenizer
import torch
from IPython.display import Audio

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create a Streamlit app
st.title("Text to Music Generation")
st.subheader("Choose the duration of the song:")

# Add a slider to choose the duration
duration_slider = st.slider("Duration (in seconds)", 1, 360, 30)

# Add a text input for the song description
song_description = st.text_input("Enter a song description:")

# Create a button to generate the song
if st.button("Generate Song"):
    # Generate the song
    unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
    audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=duration_slider * model.config.audio_encoder.frame_rate)

    # Convert the audio values to a numpy array
    audio_data = audio_values[0].cpu().numpy()

    # Display the audio
    st.audio(audio_data, sample_rate=model.config.audio_encoder.sampling_rate)