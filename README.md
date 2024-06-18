# IMAGE DENOISING PROJECT
In this project we implement the Zero DCE Framework to enhance low light images and then compare with the PIL Autocontrast Model.

# References 
[Research Paper : Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement ](https://arxiv.org/pdf/2001.06826.pdfh)
# Downloading the dataset
```
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import keras
from keras import layers

import tensorflow as tf
```

Here we download the LoL dataset file from my google drive and then unzip it.

```
!pip install gdown
import gdown

file_id = '1lNHqJmW8IJwpBbQkJC8ElfZZyBiWcDTi'
destination = 'lol_dataset.zip'

url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, destination, quiet=False)

!unzip -q lol_dataset.zip && rm lol_dataset.zip
```
Now we will create a Tensorflow Dataset 

We use 400 low-light images from the LoL Dataset training set for training, and we use the remaining low-light images for validation and a Batch size of 32. We resize the images to size 256 x 256 to be used for both training and validation.
```
SZ_IMG = 256
SZ_BATCH = 32
MAX_TRAIN_IMGS = 400

def create_dataset(image_list):
    ds = tf.data.Dataset.from_tensor_slices((image_list))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(SZ_BATCH, drop_remainder=True)
    return ds

def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(images=img, size=[SZ_IMG, SZ_IMG])
    img = img / 255.0
    return img

train_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMGS]
val_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMGS:]
test_images = sorted(glob("./lol_dataset/eval15/low/*"))


train_ds = create_dataset(train_images)
val_ds = create_dataset(val_images)

print("Training Dataset:", train_ds)
print("Validation Dataset:", val_ds)
```

# Zero DCE Framework : 
The goal of DCE-Net is to estimate a set of best-fitting light-enhancement curves (LE-curves) given an input image. The framework then maps all pixels of the input’s RGB channels by applying the curves iteratively to obtain the final enhanced image.

# Understanding light-enhancement curves
A light-enhancement curve is a kind of curve that can map a low-light image to its enhanced version automatically, where the self-adaptive curve parameters are solely dependent on the input image. When designing such a curve, three objectives should be taken into account:

Each pixel value of the enhanced image should be in the normalized range [0,1], in order to avoid information loss induced by overflow truncation.
It should be monotonous, to preserve the contrast between neighboring pixels.
The shape of this curve should be as simple as possible, and the curve should be differentiable to allow backpropagation.
The light-enhancement curve is separately applied to three RGB channels instead of solely on the illumination channel. The three-channel adjustment can better preserve the inherent color and reduce the risk of over-saturation.

![framework](https://github.com/khatriadbhut/Image_denoising_project/assets/147019819/23c9ac13-a799-4d08-afe5-480487821bdd)

# Model Architecture

The DCE-Net is a lightweight deep neural network that learns the mapping between an input image and its best-fitting curve parameter maps. The input to the DCE-Net is a low-light image while the outputs are a set of pixel-wise curve parameter maps for corresponding higher-order curves. It is a plain CNN of seven convolutional layers with symmetrical concatenation. Each layer consists of 32 convolutional kernels of size 3×3 and stride 1 followed by the ReLU activation function. The last convolutional layer is followed by the Tanh activation function, which produces 24 parameter maps for 8 iterations, where each iteration requires three curve parameter maps for the three channels.


![model_architecture](https://github.com/khatriadbhut/Image_denoising_project/assets/147019819/8709ef37-61a3-41f4-b221-9b1914ce0071)

```

def build_dce_net():
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(input_img)
    conv2 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv1)
    conv3 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv2)
    conv4 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(
        int_con3
    )
    return keras.Model(inputs=input_img, outputs=x_r)
```

#  Loss Functions : SSIM, Total variation loss, Exposure loss, Spatial Consistency Loss, Color Consistency Loss, Illumination Smoothness Loss

To enable zero-reference learning in DCE-Net, we use a set of differentiable zero-reference losses that allow us to evaluate the quality of enhanced images.
-->The color constancy loss is used to correct the potential color deviations in the enhanced image.

-->Exposure loss is to restrain under-/over-exposed regions, we use the exposure control loss. It measures the distance between the average intensity value of a local region and a preset well-exposedness level

-->Illuminattion Smoothness Loss is to preserve the monotonicity relations between neighboring pixels, the illumination smoothness loss is added to each curve parameter map.

-->The spatial consistency loss encourages spatial coherence of the enhanced image by preserving the contrast between neighboring regions across the input image and its enhanced version.

-->Structural Similarity Index (SSIM) Loss measures the similarity between two images based on luminance, contrast, and structure.

--> TV loss encourages spatial smoothness in images by penalizing the gradient differences between neighboring pixels.

```
def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = (
        mean_rgb[:, :, :, 0],
        mean_rgb[:, :, :, 1],
        mean_rgb[:, :, :, 2],
    )
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))
```
```

def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))
```
```
def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
```
```
class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(reduction="none")

        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(
            original_mean, ksize=4, strides=4, padding="VALID"
        )
        enhanced_pool = tf.nn.avg_pool2d(
            enhanced_mean, ksize=4, strides=4, padding="VALID"
        )

        d_original_left = tf.nn.conv2d(
            original_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_right = tf.nn.conv2d(
            original_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_down = tf.nn.conv2d(
            original_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool,
            self.left_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool,
            self.right_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool,
            self.down_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down
```

# Deep Curve Estimation Model

We implement the Zero-DCE framework as a Keras subclassed model.


The provided code defines a custom Keras model, ZeroDCE, for low-light image enhancement using Zero-Reference Deep Curve Estimation (DCE). The model is initialized with a DCE network built using a function build_dce_net(). The compile method sets up the optimizer and various loss trackers including spatial constancy, illumination smoothness, color constancy, exposure, SSIM, and total variation losses. The get_enhanced_image method iteratively refines the input image using the DCE network's output to produce an enhanced image. The call method performs the forward pass by processing the input through the DCE network and obtaining the enhanced image. The compute_losses method calculates and returns the different types of losses used for training. The train_step and test_step methods define custom training and evaluation steps, respectively, updating the respective loss trackers with the computed losses. Additionally, custom save_weights and load_weights methods are provided to handle saving and loading of the DCE network's weights.

```

class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.illumination_smoothness_loss_tracker = keras.metrics.Mean(
            name="illumination_smoothness_loss"
        )
        self.spatial_constancy_loss_tracker = keras.metrics.Mean(
            name="spatial_constancy_loss"
        )
        self.color_constancy_loss_tracker = keras.metrics.Mean(
            name="color_constancy_loss"
        )
        self.exposure_loss_tracker = keras.metrics.Mean(name="exposure_loss")
      #  self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
        self.ssim_loss_tracker = keras.metrics.Mean(name="ssim_loss")
        self.total_variation_loss_tracker = keras.metrics.Mean(name="total_variation_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.illumination_smoothness_loss_tracker,
            self.spatial_constancy_loss_tracker,
            self.color_constancy_loss_tracker,
            self.exposure_loss_tracker,
     #       self.perceptual_loss_tracker,
            self.ssim_loss_tracker,
            self.total_variation_loss_tracker,
        ]

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))

    #    perceptual_loss = VGGPerceptualLoss()(data, enhanced_image)
        ssim_loss = 1 - tf.image.ssim(data, enhanced_image, max_val=1.0)
        tv_loss = tf.reduce_sum(tf.image.total_variation(enhanced_image))

        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
   #         + perceptual_loss
            + ssim_loss
            + 0.1 * tv_loss
        )

        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
  #          "perceptual_loss": perceptual_loss,
            "ssim_loss": ssim_loss,
            "total_variation_loss": tv_loss,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])
 #       self.perceptual_loss_tracker.update_state(losses["perceptual_loss"])
        self.ssim_loss_tracker.update_state(losses["ssim_loss"])
        self.total_variation_loss_tracker.update_state(losses["total_variation_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        output = self.dce_model(data)
        losses = self.compute_losses(data, output)

        self.total_loss_tracker.update_state(losses["total_loss"])
        self.illumination_smoothness_loss_tracker.update_state(
            losses["illumination_smoothness_loss"]
        )
        self.spatial_constancy_loss_tracker.update_state(
            losses["spatial_constancy_loss"]
        )
        self.color_constancy_loss_tracker.update_state(losses["color_constancy_loss"])
        self.exposure_loss_tracker.update_state(losses["exposure_loss"])
#        self.perceptual_loss_tracker.update_state(losses["perceptual_loss"])
        self.ssim_loss_tracker.update_state(losses["ssim_loss"])
        self.total_variation_loss_tracker.update_state(losses["total_variation_loss"])

        return {metric.name: metric.result() for metric in self.metrics}

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

```
# Training

The below goven code initializes and compiles a ZeroDCE model for low-light image enhancement with a learning rate of 1e-4, then trains the model using the fit method with training and validation datasets (train_ds and val_ds) over 100 epochs, storing the training history in history. After training, the code defines a function plot_result that plots the training and validation losses over epochs for specified loss metrics, using matplotlib for visualization. The function takes a loss item as input, plots the respective loss values from the training history, labels the axes, adds a title, legend, and grid, and displays the plot. Finally, the code calls plot_result for each of the tracked losses, generating plots to visualize the progression of the total loss, illumination smoothness loss, spatial constancy loss, color constancy loss, exposure loss, SSIM loss, and total variation loss during training and validation.


```
# Initialize and compile the model
zero_dce_model = ZeroDCE()
zero_dce_model.compile(learning_rate=1e-4)

# Train the model
history = zero_dce_model.fit(train_ds, validation_data=val_ds, epochs=100)

# Function to plot the training and validation losses over epochs
def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

# Plot the losses
plot_result("total_loss")
plot_result("illumination_smoothness_loss")
plot_result("spatial_constancy_loss")
plot_result("color_constancy_loss")
plot_result("exposure_loss")
#plot_result("perceptual_loss")
plot_result("ssim_loss")
plot_result("total_variation_loss")
```
# Inference

Here we define two functions, plot_results and infer, for visualizing and inferring image enhancements using the ZeroDCE model. The plot_results function takes a list of images and their corresponding titles, and plots them side by side in a single row within a figure of specified size using matplotlib, turning off axis labels for a cleaner display. The infer function processes an input image for inference: it converts the image to an array, normalizes it by scaling pixel values to the range [0, 1], and adds a batch dimension. This processed image is then passed through the zero_dce_model to get the enhanced image output, which is rescaled to the range [0, 255], cast to uint8 type, converted back to a PIL image, and returned for visualization or further use.

```
def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def infer(original_image):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image

```
# Inference on test images

Now we will plot the original image, the same image with autocontrast applied using PIL (ImageOps.autocontrast), and the enhanced image side by side using plot_results. This comparison helps visualize and evaluate the improvement made by the ZeroDCE model in enhancing low-light images


```
for val_image_file in test_images:
    original_image = Image.open(val_image_file)
    enhanced_image = infer(original_image)
    plot_results(
        [original_image, ImageOps.autocontrast(original_image), enhanced_image],
        ["Original", "PIL Autocontrast", "Enhanced"],
        (20, 12),
    )
```
# Calculating PIL Value

Now, let's calculate the average Peak Signal-to-Noise Ratio (PSNR) for our model and the test dataset using calculate_psnr_for_model. It iterates through the test dataset, computes PSNR between true images and their model predictions (model(data, training=False)), and then prints the average PSNR to evaluate the model's image enhancement performance.

```
def calculate_psnr_for_model(model, test_dataset):

    psnr_values = []

    for data in test_dataset:
        true_images = data
        predicted_images = model(data, training=False)  # Get model predictions
        psnr_value = tf.image.psnr(true_images, predicted_images, max_val=1.0)
        psnr_values.append(psnr_value)

    average_psnr = tf.reduce_mean(psnr_values)
    return average_psnr.numpy()

average_psnr = calculate_psnr_for_model(zero_dce_model, test_ds)
print(f"Average PSNR for the test dataset: {average_psnr:.2f} dB")

```
