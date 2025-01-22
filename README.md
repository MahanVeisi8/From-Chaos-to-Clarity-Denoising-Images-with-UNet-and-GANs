# **Denoising Facial Emotion Dataset Using Attention U-Net and GAN**

## **Introduction**

This project presents an exploration of deep learning techniques for denoising grayscale facial images with varying levels of noise. The dataset, derived from the popular FER2013 dataset, contains pixel-based grayscale images representing facial expressions. Our objective is to remove noise from these images using advanced architectures like Attention U-Net and GANs (Generative Adversarial Networks), enabling clearer representation of the facial features.

Noise in image data can severely impact downstream tasks such as classification and detection. Here, we tackle three distinct types of noise: **low Gaussian noise**, **high Gaussian noise**, and **salt-and-pepper noise**, using state-of-the-art denoising models. 

The project is structured into three main tasks, each addressing one type of noise. Each task builds upon the previous, leveraging transfer learning and fine-tuning techniques to achieve superior performance. The models are trained and validated using the predefined splits from the FER2013 dataset (**Training**, **PublicTest**, and **PrivateTest**) to ensure robustness and generalizability.

Key contributions of this project include:
- **Attention U-Net for Denoising**: Enhanced feature extraction through attention mechanisms that adaptively focus on relevant regions of the noisy input.
- **PatchGAN Discriminator**: A patch-based approach to train GANs, ensuring both global and local denoising consistency.
- **Extensive Evaluation**: Metrics like **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index Measure)** are used to assess performance, alongside rich visualizations of denoised outputs.

By the end of this project, we aim to not only demonstrate the effectiveness of these advanced models in denoising but also provide a reusable pipeline for other denoising tasks involving similar data.

## **Setup**

### **Run This Project in Google Colab 🌟**

This notebook is pre-configured for easy execution on Google Colab, requiring **no extra setup**. All you need is:  

1. A **Google Account**.  
2. A working **internet connection**.  

Simply click the **Open in Colab** badge above and start experimenting right away! Colab will automatically install all required libraries and prepare the environment for you. 🖥️⚡  

---

## **Data Preprocessing and Noise Augmentation**

The dataset consists of grayscale images of size **48x48 pixels**, initially preprocessed to remove unnecessary information while retaining the raw pixel data for model training. Each image represents a facial expression, providing valuable insights for denoising tasks. We performed the following steps for data preparation:

1. **Loading and Preprocessing**: The images were extracted from the FER2013 dataset and preprocessed to maintain uniformity in size and pixel intensity normalization. 
2. **Splitting**: The dataset was split into predefined **training**, **validation**, and **test** sets for consistency during experimentation.

### **Sample Images**
Below are some examples of the raw grayscale images from the dataset:

![Sample Images](assets/samples.png)

---

To study the denoising effects systematically, we introduced **three distinct types of noise** into the images, mimicking real-world scenarios of image corruption. These noise augmentations help evaluate the robustness of the denoising models:

1. **Low Gaussian Noise**: Mild Gaussian noise with a standard deviation of 0.2 and noise factor of 0.2.
2. **High Gaussian Noise**: Intense Gaussian noise with a standard deviation of 0.4 and noise factor of 0.3.
3. **Salt-and-Pepper Noise**: A random noise pattern with a noise factor of 0.1 and an equal mix of "salt" (white) and "pepper" (black) pixels.

### **Sample Images with Noise Augmentations**
#### Gaussian Noise (Low)
![Low Gaussian Noise](assets/LowGaussianNoise.png)

#### Gaussian Noise (High)
![High Gaussian Noise](assets/HighGaussianNoise.png)

#### Salt-and-Pepper Noise
![Salt-and-Pepper Noise](assets/Salt-and-PepperNoise.png)
