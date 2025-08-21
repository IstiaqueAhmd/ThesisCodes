#import necessary libraries
import os
import numpy as np
import scipy.io as sio
import pywt
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""
This code generates In-phase, Quadrature and Amplitude scalograms using raw I/Q samples. 
"""

def generateWaveletTransform(data_type):
    input_dir = f'Data/ModClassDataFiles/{data_type}'
    output_dir = f'Data/Scalograms/{data_type}'
    samples_dir = f'Data/Samples/{data_type}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    sample_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            mat_data = sio.loadmat(os.path.join(input_dir, filename))
            data = mat_data['frame'].flatten()

            I = np.real(data)
            Q = np.imag(data)
            amplitude = np.sqrt(I ** 2 + Q ** 2)

            wavelet = 'cmor1.5-0.5'
            scales = np.logspace(0.5, 2, num=200)

            def compute_cwt(signal):
                coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / 1000)
                coeffs = np.abs(coeffs)
                scaler = MinMaxScaler()
                return scaler.fit_transform(coeffs)

            # Compute wavelet transforms
            cwt_I = compute_cwt(I)
            cwt_Q = compute_cwt(Q)
            cwt_amplitude = compute_cwt(amplitude)

            # Resize to 224x224
            cwt_I = cv2.resize(cwt_I, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cwt_Q = cv2.resize(cwt_Q, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cwt_amplitude = cv2.resize(cwt_amplitude, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            # Stack into a 3-channel image
            stacked_scalogram = np.stack([cwt_I, cwt_Q, cwt_amplitude], axis=-1)

            # Save .npy file
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
            np.save(output_path, stacked_scalogram)

            # Optionally save sample images
            if sample_count < 5:
                i_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_i.png")
                q_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_q.png")
                amp_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_amp.png")

                plt.imsave(i_img_path, cwt_I, cmap='gray', vmin=0, vmax=1)
                plt.imsave(q_img_path, cwt_Q, cmap='gray', vmin=0, vmax=1)
                plt.imsave(amp_img_path, cwt_amplitude, cmap='gray', vmin=0, vmax=1)

                sample_count += 1

    print(f"3-channel CWT scalograms (I, Q, Amplitude) saved for {data_type}, with {sample_count} sample images.")

# Run for desired modulation types
classes = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK", "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]

for data_type in classes:
    generateWaveletTransform(data_type)
