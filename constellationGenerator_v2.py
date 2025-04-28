import os
import numpy as np
import scipy.io as sio
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def generateConstellationDiagram(data_type):
    input_dir = f'Data/ModClassDataFiles/{data_type}'
    output_dir = f'Data/Constellations/{data_type}'
    samples_dir = f'Data/Samples(Constellations)/{data_type}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    sample_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            mat_data = sio.loadmat(os.path.join(input_dir, filename))
            data = mat_data['frame'].flatten()

            I = np.real(data)
            Q = np.imag(data)

            # Generate constellation diagram (2D histogram)
            hist, x_edges, y_edges = np.histogram2d(I, Q, bins=100)
            hist = hist.T  # Transpose to align I and Q correctly

            # Normalize histogram to [0, 1]
            hist_min, hist_max = hist.min(), hist.max()
            hist_range = hist_max - hist_min
            if hist_range == 0:
                hist_normalized = np.zeros_like(hist, dtype=np.float32)
            else:
                hist_normalized = (hist - hist_min) / hist_range

            # Resize constellation diagram
            constellation = cv2.resize(hist_normalized, (224, 224), interpolation=cv2.INTER_LANCZOS4)


            # Save stacked image
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.npy')
            np.save(output_path, constellation)

            # Save sample images for first 5 files
            if sample_count < 5:
                # Save components separately
                const_path = os.path.join(samples_dir, f"{os.path.splitext(filename)[0]}_constellation.png")
                plt.imsave(const_path, constellation, cmap='gray', vmin=0, vmax=1)
                sample_count += 1

    print(f"Processed {data_type}")


# Process all modulation classes
classes = ["16QAM", "64QAM", "8PSK", "B-FM", "BPSK",
           "CPFSK", "DSB-AM", "GFSK", "PAM4", "QPSK", "SSB-AM"]
for data_type in classes:
    generateConstellationDiagram(data_type)
