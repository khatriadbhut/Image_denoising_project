import os
from glob import glob
import random
from train.training import train_model,charbonnier_loss,peak_signal_noise_ratio
from inference.infer import infer_and_save
import requests
import zipfile
import io
import tensorflow as tf


IMAGE_SIZE = 128
BATCH_SIZE = 4
MAX_TRAIN_IMAGES = 300
EPOCHS = 50
MODEL_SAVE_PATH = "mirnet_model.h5"

random.seed(10)

# Download and unzip dataset
def download_and_extract(url, extract_to='.'):
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)

download_and_extract('https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip')

train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
train_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[:MAX_TRAIN_IMAGES]

val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
val_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[MAX_TRAIN_IMAGES:]

test_low_light_images = sorted(glob("./test/low/*"))


history, model = train_model(train_low_light_images, train_enhanced_images, val_low_light_images, val_enhanced_images, IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH)

# Plot training history
import matplotlib.pyplot as plt

def plot_history(history, value, name):
    plt.plot(history.history[value], label=f"train_{name.lower()}")
    plt.plot(history.history[f"val_{value}"], label=f"val_{name.lower()}")
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(f"Train and Validation {name} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

plot_history(history, "loss", "Loss")
plot_history(history, "peak_signal_noise_ratio", "PSNR")


# Infer and save results
infer_and_save(model, test_low_light_images, "./test/predicted")
