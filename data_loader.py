import numpy as np
from matplotlib import pyplot as plt


def load(name):
    input_images = "./data/" + name
    data = np.load(input_images)  # 28x28 (sound familiar?) grayscale bitmap in numpy .npy format; images are centered
    data = data / 255
    data = np.reshape(data, (data.shape[0], 28, 28, 1))  # fourth dimension is color
    return data

def test_loader(data_set):
    data = load(data_set)
    img_w, img_h = data.shape[1:3]
    print(img_w, img_h)
    print(data.shape)
    plt.imshow(data[8375, :, :, 0], cmap='Greys')
    plt.show()

# test_loader()

