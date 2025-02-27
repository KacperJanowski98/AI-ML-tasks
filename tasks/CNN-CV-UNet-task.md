## CNN Training

### **Introduction**

The aim of the project is to develop an efficient AI pipeline for the segmentation of anatomical structures in ultrasound (US) images stored in DICOM format. The solution should correctly identify selected structures (e.g., organs, pathological changes) in the images and be optimized for fast inference on devices with limited computational power (edge computing). To achieve this, a neural network model with a U-Net architecture was designed for segmentation along with a series of preprocessing and optimization steps. The key elements of the solution are presented below: data acquisition and preparation, model architecture, training, inference optimization, quality assessment metrics, testing, and deployment methods.

### **DICOM Data Acquisition**

In the first step, sample data—US images in DICOM format—were gathered. Public sources such as The Cancer Imaging Archive (TCIA) or available datasets on Kaggle, which include ultrasound images with annotations, were used. For example, TCIA provides medical datasets (including US) devoid of sensitive data, ready for download and use in research. If ready-made reference masks were not available, own masks could be prepared by manually annotating several images or using semi-automatic methods. The pydicom library was used for reading DICOM files, which allows reading DICOM headers and extracting pixel data using the pixel_array property. This makes it easy to obtain a pixel intensity array from a US image and convert it to a numpy format for further processing.

### **Example of Reading DICOM in Python:**

```Python
import pydicom
ds = pydicom.dcmread("example_file.dcm")
image = ds.pixel_array  # numpy array with US image pixels
```

After loading the data, it was divided into training, validation, and test sets (e.g., 70/15/15%). This allows the model to train and evaluate its quality on previously unseen images. If the number of images is small, applying augmentation (e.g., flips, slight rotations, noise addition) is recommended to increase the diversity of the training data.

### **Data Preparation (Preprocessing)**

Ultrasound images require proper preprocessing to facilitate the work of the segmenting model. In particular, ultrasound suffers from characteristic speckle noise, which reduces contrast and resolution. Therefore, the following preprocessing steps were applied:

• **Intensity Normalization** – Conversion of pixel data to a unified scale (e.g., 0-1). For grayscale US images (typically 8- or 16-bit), min-max normalization or subtracting the mean and dividing by the standard deviation was performed. Normalization facilitates model training by making data comparable across images.

• **Noise Reduction (Denoising)** – To reduce speckle noise, filtering was applied. The simplest approach is a median or Gaussian filter to smooth out minor artifacts. More advanced methods include anisotropic diffusion or specialized speckle filters (e.g., Lee or Frost filter), but given limited time, a 3x3 median was chosen, which effectively reduces minor spots without significant loss of edge sharpness. Noise reduction is crucial because speckle noise lowers diagnostic quality and complicates the segmentation of important structures.

• **Contrast Enhancement**– To highlight anatomical structures, image contrast was increased. CLAHE (Contrast Limited Adaptive Histogram Equalization) was used, available in OpenCV. CLAHE enhances local contrast in medical images without overly amplifying noise. This makes the boundaries of structures (e.g., organ contours) more visible to the model.

• **Size Scaling/Cropping** – If needed, images were scaled to a resolution required by the model (e.g., 256x256 or 512x512). U-Net requires a fixed input dimension, so if original DICOMs had different resolutions, scaling or cropping of the central square was performed. Structures' proportions were preserved to avoid distorting segmented objects.

These steps were implemented using the OpenCV library (e.g., cv2.medianBlur, cv2.createCLAHE) and numpy functions. After preprocessing, image data was prepared for training and inference—cleaned of excessive noise and normalized, which should improve segmentation efficiency.

### **Segmentation Model Architecture (U-Net)**

The U-Net architecture—a convolutional neural network in an encoder-decoder layout with skip connections—was chosen for segmentation. U-Net is a classic for medical image segmentation, designed for this purpose. Its name comes from its U-shaped form. It consists of an encoding part (subsequent convolutional layers and pooling reducing feature size, extracting context) and a decoding part (transposed convolutions/upsampling increasing resolution and restoring details). An important element is skip connections—connections between corresponding encoder and decoder levels, passing high-resolution features omitted during downsampling. This allows the model to learn both global image information and local details crucial for precise boundary determination of segmented objects.

**Justification for Choosing U-Net:** The U-Net architecture has achieved great success in biomedical segmentation—dominating the ISBI Cell Tracking Challenge in 2015, significantly outperforming earlier methods. U-Net allows segmenting even high-resolution images (e.g., 512x512) in reasonable time using fully convolutional operations (FCN). Its phenomenal effectiveness has led to many modifications (U-Net++, Attention U-Net, 3D U-Net), but the base version often suffices for accurate anatomical structure segmentation. U-Net's advantage is also its relatively small model size compared to very deep segmentation networks, which favors deployment on edge devices. For these reasons, U-Net is a safe and proven choice for this task. An alternative approach (if data were very limited) could involve using pre-trained models (transfer learning) or transformer-based segmentation models, but due to simplicity and effectiveness, U-Net was implemented.

**Model Implementation:** The model was implemented in the PyTorch library. 2D convolutions, ReLU layers, and batch normalization were used in the encoder block, followed by deconvolutional layers (ConvTranspose2d) in the decoder. The architecture was defined modularly—for example, the UNet class contains internal classes/modules for each encoder and decoder level, improving readability. Pseudocode for U-Net construction:

```Python
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        # Encoder
        self.down1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(features[0], features[1])
        # ... analogous blocks further ...
        # Decoder
        self.up1 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[3], features[2])
        # ... further decoder levels ...
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        # Data flow through encoder
        skip1 = self.down1(x); x = self.pool1(skip1)
```
