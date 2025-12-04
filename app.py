import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from scipy.signal import butter, filtfilt
from streamlit_folium import st_folium

st.title("Askelmittaus ja GPS-analyysi")

BASE_DIR = os.path.dirname(__file__)
df_accel = pd.read_csv("Accelerometer.csv")
df_gps = pd.read_csv("Location.csv")

time = df_accel["Time (s)"].values
y = df_accel["Y (m/s^2)"].values

idx = np.where(time >= 20)[0][0]
time = time[idx:] - time[idx]
y = y[idx:]

dt = time[1] - time[0]
fs = 1/dt
nyq = fs/2
cutoff = 2
b,a = butter(3, cutoff/nyq, btype='low')
data_filt = filtfilt(b,a,y)

lat = df_gps["Latitude (°)"].values
lon = df_gps["Longitude (°)"].values
gpstime = df_gps["Time (s)"].values

idx2 = np.where(gpstime >= 20)[0][0]

lat = lat[idx2:]
lon = lon[idx2:]
gpstime = gpstime[idx2:] - gpstime[idx2]

R = 6371000

lat_r = np.radians(lat)
lon_r = np.radians(lon)

dist = 0
for i in range(len(lat_r)-1):
    dlat = lat_r[i+1] - lat_r[i]
    dlon = lon_r[i+1] - lon_r[i]
    a = np.sin(dlat/2)**2 + np.cos(lat_r[i])*np.cos(lat_r[i+1])*np.sin(dlon/2)**2
    dist += 2 * R * np.arcsin(np.sqrt(a))

duration = gpstime.max()
speed = dist / duration

st.write("Kuljettu matka (m):",round(dist  , 2))
st.write("Keskinopeus (m/s):", round(speed  , 2))

data_raw = y

steps_zc = 0
threshold = 0.2 
for i in range(len(data_raw) - 1):
    if (data_raw[i] * data_raw[i + 1] < 0) and (abs(data_raw[i]) > threshold or abs(data_raw[i+1]) > threshold):
        steps_zc += 1


steps_zc //= 2
st.write("Askelmäärä (suodatettu):", steps_zc)


N = len(data_raw)
dt = time[1] - time[0]
T_tot = time[-1] - time[0]

fourier = np.fft.fft(data_raw)
psd = np.abs(fourier)**2 / N
freq = np.fft.fftfreq(N, d=dt)


mask = (freq > 0.8) & (freq < 3.0)
freq_pos = freq[mask]
psd_pos = psd[mask]

peak_freq = freq_pos[np.argmax(psd_pos)]
steps_fft = int(peak_freq * T_tot)
st.write("Askelmäärä fourier:", steps_fft)

st.subheader("Suodatettu Y")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(time, data_filt)
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Kiihtyvyys (m/s²)")
st.pyplot(fig)





st.subheader("Tehospektri")
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(freq_pos, psd_pos)
ax2.set_xlabel("Taajuus [Hz]")
ax2.set_ylabel("Teho")
st.pyplot(fig2)

lat = df_gps["Latitude (°)"].values
lon = df_gps["Longitude (°)"].values
gpstime = df_gps["Time (s)"].values
idx2 = np.where(gpstime >= 60)[0][0]
lat = lat[idx2:]
lon = lon[idx2:]
gpstime = gpstime[idx2:] - gpstime[idx2]

st.subheader("GPS-reitti kartalla")
m = folium.Map(location=[lat[0], lon[0]], zoom_start=16)
folium.PolyLine(list(zip(lat, lon)), color="red", weight=3).add_to(m)
st_folium(m, width=700, height=500)
