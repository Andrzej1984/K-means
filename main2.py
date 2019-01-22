import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import time


from pydicom.data import get_testdata_files


filename = "MyHead\MR000058.dcm"
ds = pydicom.dcmread(filename)

ds.PixelData
ds.pixel_array
ds.pixel_array.shape

img =ds.pixel_array

#rozmywanie
dst = cv2.blur(img,(5,5))
plt.imshow(dst, cmap="gray")
plt.show()


Z= np.float32(dst)
Z= Z.reshape((-1, 1))

t_start = time.clock()#włączam stoper

#algorytm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 10.0)

K=6
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res3 = center[label.flatten()]
res3 = res3.reshape((img.shape))
t_end = time.clock()

t = t_end - t_start#wyłączam stoper
print(t)

component = np.zeros(img.shape, np.uint8)
label2 = label.reshape((img.shape[0], img.shape[1]))
component[label2 == 2 ] = img[label2 == 2]
component[label2 == 0 ] = img[label2 == 0]
#component[label2 == 3 ] = img[label2 == 3]

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")

plt.subplot(1, 2, 2)

component[label2 == 2 ] = img[label2 == 2]
plt.imshow(component, cmap="gray")
plt.show()

plt.subplot(1, 2, 1)
component = np.zeros(img.shape, np.uint8)
component[label2 == 0 ] = img[label2 == 0]
plt.imshow(component, cmap="gray")
plt.subplot(1, 2, 2)
component = np.zeros(img.shape, np.uint8)
component[label2 == 2 ] = img[label2 == 2]
plt.imshow(component, cmap="gray")
plt.show()


for i in range(6):
    label2 = label.reshape((img.shape[0], img.shape[1]))
    component = np.zeros(img.shape, np.uint8)
    component[label2 == i] = img[label2 == i]

    plt.subplot(3, 2, i + 1)
    plt.imshow(component, cmap="gray")

    text="warstwa"
    plt.title(i)
    plt.xticks([])
    plt.yticks([])



plt.show()
component = np.zeros(img.shape, np.uint8)


#print(ds.pixel_array)

