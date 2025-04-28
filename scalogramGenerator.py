import os
import numpy as np
import scipy.io as sio
import pywt
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
            phase = np.arctan2(Q, I)

            wavelet = 'cmor2.5-1.5'
            scales = np.logspace(0.5, 2, num=200)

            def compute_cwt(signal):
                coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / 1000)
                coeffs = np.abs(coeffs)
                scaler = MinMaxScaler()
                return scaler.fit_transform(coeffs)

            cwt_amplitude = compute_cwt(amplitude)
            cwt_phase = compute_cwt(phase)

            cwt_amplitude = cv2.resize(cwt_amplitude, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cwt_phase = cv2.resize(cwt_phase, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            stacked_scalogram = np.stack([cwt_amplitude, cwt_phase], axis=-1)

            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
            np.save(output_path, stacked_scalogram)

            if sample_count < 5:
                amp_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_amp.png")
                phase_img_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_phase.png")

                plt.imsave(amp_img_path, cwt_amplitude, cmap='gray', vmin=0, vmax=1)
                plt.imsave(phase_img_path, cwt_phase, cmap='gray', vmin=0, vmax=1)

                sample_count += 1
    print(f"Stacked wavelet transforms saved for {data_type}, including 5 raw sample images.")

# Run for multiple modulation types
classes = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK", "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]
for data_type in classes:
    generateWaveletTransform(data_type)

