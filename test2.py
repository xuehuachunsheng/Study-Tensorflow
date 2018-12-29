
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
path = '/Users/wuyanxue/Desktop/img_00000001.jpg'

img = Image.open(path).convert('RGB')

array = np.array(img)
a = [72, 79, 232, 273]
# ymin:ymax, xmin:xmax
array = array[a[1]:a[3], a[0]:a[2], :]
plt.imshow(Image.fromarray(array, mode='RGB'))
plt.show()

