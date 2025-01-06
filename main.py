import cv2
import numpy as np
import matplotlib.pyplot as plt

def dark_channel(image):
    """
    Calculate the dark channel of the image.
    The dark channel is the minimum intensity across the RGB channels in a local patch.
    """
    patch_size = 15  # Typical patch size for dark channel calculation
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimating_atmospheric_light(image, dark_channel):
    """
    Estimate the atmospheric light by finding the brightest pixels in the dark channel.
    """
    num_brightest = int(0.001 * dark_channel.size)  # Top 0.1% brightest pixels
    flat_dark_channel = dark_channel.ravel()
    flat_image = image.reshape((-1, 3))
    indices = np.argsort(flat_dark_channel)[-num_brightest:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)
    return atmospheric_light

def transmission_estimate(image, atmospheric_light):
    """
    Estimate the transmission map using the dark channel and atmospheric light.
    """
    omega = 0.95  # Typical omega value for haze removal
    normalized_image = image / atmospheric_light
    transmission = 1 - omega * dark_channel(normalized_image)
    return np.clip(transmission, 0.1, 1)  # Avoid complete blackness

def recovering_scene_radiance(image, atmospheric_light, transmission):
    """
    Recover the scene radiance (dehazed image) using the transmission map and atmospheric light.
    """
    transmission = np.maximum(transmission, 0.1)  # Avoid division by zero
    transmission = transmission[:, :, np.newaxis]  # Expand transmission to 3 channels
    J = (image - atmospheric_light) / transmission + atmospheric_light
    return np.clip(J, 0, 1)

def guided_filter(transmission, image, window_size):
    """
    Refine the transmission map using guided filtering.
    """
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    guided = cv2.ximgproc.createGuidedFilter(gray_image, radius=window_size // 2, eps=1e-3)
    refined_transmission = guided.filter(transmission.astype(np.float32))
    return refined_transmission

# Main Program
if __name__ == "__main__":
    # Load input hazy image
    hazy_image = cv2.imread("12.jpg")
    hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)
    hazy_image = hazy_image.astype(np.float32) / 255.0

    # Display input image
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(hazy_image)
    plt.title("Hazy Image - Input")

    # Finding Dark Channel
    J_DARK = dark_channel(hazy_image)
    plt.subplot(2, 3, 2)
    plt.imshow(J_DARK, cmap="gray")
    plt.title("Dark Channel")

    # Finding Atmospheric Light
    A = estimating_atmospheric_light(hazy_image, J_DARK)

    # Finding Transmission
    t = transmission_estimate(hazy_image, A)
    plt.subplot(2, 3, 3)
    plt.imshow(t, cmap="gray")
    plt.title("Transmission Estimate")

    # Finding Haze-Free Image
    J = recovering_scene_radiance(hazy_image, A, t)
    plt.subplot(2, 3, 4)
    plt.imshow(np.clip(J, 0, 1))
    plt.title("Recovered Image")

    # Finding Refined Transmission using Guided Filter
    window_size = 75
    FG = guided_filter(t, hazy_image, window_size)
    plt.subplot(2, 3, 5)
    plt.imshow(FG, cmap="gray")
    plt.title("Refined Transmission Estimate using Guided Filter")

    # Finding Refined Haze-Free Image
    J_Refined = recovering_scene_radiance(hazy_image, A, FG)
    plt.subplot(2, 3, 6)
    plt.imshow(np.clip(J_Refined, 0, 1))
    plt.title("Refined Recovered Image")

    # Show all results
    plt.tight_layout()
    plt.show()
