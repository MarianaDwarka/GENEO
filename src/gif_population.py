from utils import plot_map_lazy_frame
import matplotlib.pyplot as plt
import streamlit as st
import polars as pl
import imageio.v2 as imageio
import os
from PIL import Image
import time

st.set_page_config(layout="wide")

if not os.path.exists("gif_p"):
    os.makedirs("gif_p")


def show_frames_sequence():
    frames = []
    for i in range(24):
        df = pl.scan_csv(f"csv's/Hora_0{i}_MEX_v2.csv" if i<10 else f"csv's/Hora_{i}_MEX_v2.csv")
        filename = f'plot_{i}.png'
        full_path = os.path.join("gif_p", filename)
        plot_map_lazy_frame(df, filename)
        frames.append(full_path)
    
    return frames


frames = show_frames_sequence()
    
placeholder = st.empty()
play_button = st.button("Reproducir/Pausar")
    
if 'playing' not in st.session_state:
    st.session_state.playing = False
    st.session_state.frame_idx = 0
    
if play_button:
    st.session_state.playing = not st.session_state.playing
    
if st.session_state.playing:
    for i in range(24):
        st.session_state.frame_idx = i
        placeholder.image(frames[i], caption=f'Distribución - Hora {i}', use_container_width=True)
        time.sleep(0.5)  
else:
    placeholder.image(frames[st.session_state.frame_idx], 
                        caption=f'Distribución - Hora {st.session_state.frame_idx}', 
                        use_container_width=True)