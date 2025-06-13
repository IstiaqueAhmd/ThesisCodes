import os
import numpy as np
import scipy.io as sio
import pywt
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def generateWaveletTransform(data_type):
    input_dir = f'ModClassDataFiles/{data_type}'
    output_dir = f'Scalograms(I-Q)/{data_type}'
    samples_dir = f'Samples(I-Q)/{data_type}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    sample_count = 0


    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            mat_data = sio.loadmat(os.path.join(input_dir, filename))
            data = mat_data['frame'].flatten()

            I = np.real(data)
            Q = np.imag(data)

            wavelet = 'cmor2.5-1.5'
            scales = np.logspace(0.5, 2, num=200)

            def compute_cwt(signal):
                coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / 1000)
                coeffs = np.abs(coeffs)
                scaler = MinMaxScaler()
                return scaler.fit_transform(coeffs)

            cwtI = compute_cwt(I)
            cwtQ = compute_cwt(Q)

            cwtI = cv2.resize(cwtI, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            cwtQ = cv2.resize(cwtQ, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            stacked_scalogram = np.stack([cwtI, cwtQ], axis=-1)

            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
            np.save(output_path, stacked_scalogram)

            if sample_count < 5:
                i_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_inphase.png")
                q_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_quadrature.png")

                plt.imsave(i_path, cwtI, cmap='gray', vmin=0, vmax=1)
                plt.imsave(q_path, cwtQ, cmap='gray', vmin=0, vmax=1)

                sample_count += 1
    print(f"Stacked wavelet transforms saved for {data_type}, including 5 raw sample images.")

# Run for multiple modulation types
classes = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK", "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]
for data_type in classes:
    generateWaveletTransform(data_type)

