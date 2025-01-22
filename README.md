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

### **Models Overview**

This project explores two powerful architectures for image denoising: **Attention U-Net** and **PatchGAN**. These models are specifically tailored to handle noisy grayscale facial images, enabling effective restoration of visual features. Below is a detailed explanation of their structure and design choices, supported by code.

---

### **1. Attention U-Net**

The **Attention U-Net** builds upon the classic U-Net architecture by incorporating **attention mechanisms**, enabling the model to focus on relevant regions of the input dynamically. This enhancement ensures effective noise suppression while preserving essential structural and contextual features, making it highly suitable for image denoising tasks.

#### **Architecture**
The Attention U-Net is divided into some components:

```python
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_attention=True, debug=False):
        super(AttentionUNet, self).__init__()
        self.debug = debug
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 16)
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)
        self.enc4 = EncoderBlock(64, 128)
        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)
        # Decoder
        self.dec4 = DecoderBlock(256, 128, use_attention=use_attention, debug=debug)
        self.dec3 = DecoderBlock(128, 64, use_attention=use_attention, debug=debug)
        self.dec2 = DecoderBlock(64, 32, use_attention=use_attention, debug=debug)
        self.dec1 = DecoderBlock(32, 16, use_attention=False, debug=debug)
        # Final Output
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
```

1. **Encoder**: 
   - Each `EncoderBlock` consists of convolutional layers for feature extraction and max-pooling for downsampling.
   - Optional **attention modules** refine features by focusing on spatially important regions based on the input context.

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, stride=2, padding=0, debug=False):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride, padding=padding)
        if use_attention:
            self.attention = AttentionBlock(out_channels, out_channels, out_channels)
```

2. **Bottleneck**: 
   - A dense `ConvBlock` bridges the encoder and decoder, aggregating global context to capture high-level features.

3. **Decoder**:
   - Each `DecoderBlock` upsamples the feature maps using transposed convolutions, enabling reconstruction at higher resolutions.
   - Skip connections integrate fine-grained details from the encoder for precise restoration.
   - Attention mechanisms selectively refine the reconstructed features, helping prioritize meaningful information.

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, debug=False):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
        if use_attention:
            self.attention = AttentionBlock(out_channels, out_channels, out_channels)
```

4. **Output Layer**:
   - A single convolutional layer reduces the feature map to the target image dimensions, reconstructing the output to match the original image size (e.g., \(1 \times 48 \times 48\)).

---

#### **Training Highlights**
The Attention U-Net was trained using a carefully designed configuration:

- **Loss Function**: Mean Squared Error (MSE) ensures pixel-wise consistency between the denoised output and the clean ground truth. This choice balances simplicity and effectiveness for grayscale image restoration.
  
- **Optimization**:
  - **Optimizer**: Adam optimizer with an initial learning rate of `1e-3` ensures fast convergence.
  - **Scheduler**: A ReduceLROnPlateau scheduler dynamically lowers the learning rate when validation loss stagnates, preventing overfitting and improving generalization.

---

### **2. PatchGAN**

The **PatchGAN** discriminator introduces a **Generative Adversarial Network (GAN)** structure. It evaluates denoised outputs on a **patch level**, ensuring both global consistency and local accuracy.

#### **Generator**
The generator directly reuses the **Attention U-Net**, leveraging its attention mechanisms for precise denoising.

#### **Discriminator**
The discriminator evaluates the denoised image by dividing it into smaller patches and assigning a **real** or **fake** score to each patch.

```python
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, stride=[2, 2, 2, 2, 2, 2], padding=[0, 0, 0, 0, 0, 0], use_fc=False, global_pooling=False, debug=False):
        super(PatchGANDiscriminator, self).__init__()
        self.debug = debug
        self.use_fc = use_fc
        self.global_pooling = global_pooling

        # Encoder layers
        self.enc1 = EncoderBlock(in_channels, base_channels, use_attention=True, stride=stride[0], padding=padding[0], debug=debug)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, use_attention=False, stride=stride[1], padding=padding[1], debug=debug)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, use_attention=True, stride=stride[2], padding=padding[2], debug=debug)
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels * 2, 1, kernel_size=2, stride=stride[5], padding=padding[5])

        # Fully connected layers
        if self.use_fc:
            self.fc_dim = 12 * 12
            self.fc = nn.Sequential(
                nn.Linear(base_channels * 2, self.fc_dim), 
                nn.Tanh(),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh()
            )
```

1. **Inputs**:
   - The discriminator takes two inputs:
     - **Noisy Image** (real or generated).
     - **Clean Image** (ground truth or generated).

   These are concatenated channel-wise for evaluation.

2. **Patch-Level Predictions**:
   - Using encoder blocks, the discriminator extracts features and downsamples the input.
   - The final layer outputs patch-based predictions, where each patch represents the discriminator’s confidence in the denoised region.

3. **Label Smoothing**:
   - For stability, **real patches** are labeled as **0.9**, and **fake patches** as **0.1**, instead of the traditional binary values (1 and 0).

```python
def forward(self, x, y):
    combined = torch.cat([x, y], dim=1)
    features, _ = self.enc1(combined)
    features, _ = self.enc2(features)
    out = self.final_conv(features)  # Patch-based output
    return out
```

#### **Key Features**
- **Patch-Based Output**: Ensures spatial consistency in the denoised image.
- **Attention Mechanism**: Selectively refines feature extraction in the discriminator.
- **Label Smoothing**: Encourages stable adversarial training by avoiding overly confident predictions.

