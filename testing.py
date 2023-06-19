import cv2
import numpy as np

image = cv2.imread('dog.png')

noise = np.random.normal(loc = 0, scale = 3, size = image.shape).astype(np.uint8)
image = cv2.add(image, noise)

cv2.imwrite('dog_noise.png', image)