import io

import matplotlib.pyplot as plt
import librosa
import numpy as np
import IPython
from PIL import Image


def visualize_mels(mels, sr=16000, from_db=False, fps=25, save=False, truth=False):
    str_truth = "Ground Truth" if truth else "Prediction"
    frame_length = sr // fps
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Display the colored mel spectrogram on the left subplot
    # print(torch.is_tensor(mels))
    # if torch.is_tensor(mels):
    #     mels = mels.detach().numpy()
    # mels += 1e-8
    if not from_db:
        mels = librosa.power_to_db(mels, ref=np.max)

    librosa.display.specshow(mels,
                             y_axis='mel', fmax=sr / 2.0, n_fft=frame_length,
                             x_axis='time', ax=axs[0])
    axs[0].set_title(f'Mel spectrogram - {str_truth}')

    # Display the black and white mel spectrogram on the right subplot
    bw = plt.imshow(mels, cmap='gray_r')
    axs[1].set_title(f'Mel spectrogram (black and white) - {str_truth}')
    axs[1].set_xlabel("Time")
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    # Add a colorbar to the right subplot
    fig.colorbar(axs[0].collections[0], ax=axs[0], format='%+2.0f dB')
    # fig.colorbar(bw, ax=axs[1])
    plt.tight_layout()

    mel_pic = None
    if save:
        with io.BytesIO() as out:
            fig.savefig(out, format="png")  # Add dpi= to match your figsize
            pic = Image.open(out)
            mel_pic = np.array(pic.getdata(), dtype=np.uint8).reshape(pic.size[1], pic.size[0], -1)
    else:
        plt.show()
    plt.close(fig)
    return mel_pic

def visualize_latent(y, gt=False, save=False, vmin=-1, vmax=1, desc="Ground Truth"):
    # title = "Ground Truth" if gt else "Prediction"
    # Create random tensor of size (8x118x16)
    tensor = y.detach().cpu().squeeze().numpy()    
    if len(tensor.shape) == 1:
        # If the tensor is 1D, create a line plot
        # print("Tensor", tensor)
        fig, ax = plt.subplots()
        ax.plot(tensor)
        ax.set_title(f"Latent Tensor {desc}")
    elif len(tensor.shape) == 2:
        # If the tensor is 1D, create a line plot
        fig, ax = plt.subplots()
        ax.imshow(tensor)
        ax.set_title(f"Latent Tensor {desc}")
    else:
        # print("SHAPE TENSOR:",tensor.shape)
        num_channels = tensor.shape[0]
        # Determine the number of rows and columns for the subplots
        num_cols = min(num_channels, 4) # Set the number of columns to 4 or less
        num_rows = int(np.ceil(num_channels / num_cols))
        
        # Iterate over the tensor and plot each 2D array
        # Create 8 subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 6))
            # Flatten the axs array to make it easier to iterate over
        axs = axs.flatten() 
        for i in range(num_channels):
            # print("LATENT:", tensor[i].max(), tensor[i].min())
            im = axs[i].imshow(tensor[i], cmap='viridis', vmin=vmin, vmax=vmax)
            axs[i].set_title('Array {}'.format(i+1))
            fig.colorbar(im, ax=axs[i])
        

    # Show the plot
    latent_pic = None
    if save:
        with io.BytesIO() as out:
            fig.savefig(out, format="png")  # Add dpi= to match your figsize
            pic = Image.open(out)
            latent_pic = np.array(pic.getdata(), dtype=np.uint8).reshape(pic.size[1], pic.size[0], -1)
    else:
        plt.show()
    plt.close(fig)
    return latent_pic
    

def play_audio(path):
    IPython.display.Audio(path)

def compare_audio(y_before, y_after, sr = 16000, desc=""):
    print(len(y_before))
    time_before = np.arange(len(y_before)) / float(sr)
    time_after = np.arange(len(y_after)) / float(sr)
    figure, axis = plt.subplots(1,2, figsize=(15, 3), sharey=True)
  
    axis[0].plot(time_before, y_before)
    axis[0].set_ylabel("Amplitude")
    axis[0].set_xlabel("Time")
    axis[0].set_title("Original Audio")

    axis[1].plot(time_after, y_after)
    axis[1].set_ylabel("Amplitude")
    axis[1].set_xlabel("Time")
    axis[1].set_title(f"{desc} Audio")
    
    plt.show()
    
def visualize_audio(y, sample_rate=16000, desc="Audio"):
    # Generate the time array for the audio clip
    time = np.arange(len(y)) / float(sample_rate)

    # Create the plot
    plt.figure(figsize=(8, 3))
    plt.plot(time, y)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")  # now the x-axis represents time in seconds
    plt.title(f"{desc}")
    plt.show()
