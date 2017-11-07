import tifffile as tiff
import numpy as np
from Unet import get_unet, jaccard_coef, jaccard_coef_int
ISZ = 256
N_Cls = 2
smooth = 1e-12

#print('start load...')
#data = np.load('data_swp.npy')
#label = np.load('label_swp.npy')
#print('load complete')

img = tiff.imread('trees/6120_2_2_2-3_crops.tif')
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)

img = img.astype(np.float32)
img = img / 255
print(img.shape)
print(type(img[0,0,0]))
data = img.reshape(1, 3, 256, 256)
print(data.shape)
print(type(data[0,0,0,0]))


model = get_unet()
model.load_weights('unet_weight_ep28_BS10__unet_CH8L.hdf5')
output = model.predict(data, batch_size=4, verbose=1)
tiff.imsave('mask_pred_trees.tif', output[0])
print(type(output))
print(output.shape)
'''
print(type(output[0]))
print(255 * output[0])
a = output[0].astype(np.uint8)
print('output.shape', a)
tiff.imsave('mask_pred.tif', a)
tiff.imsave('mask_output[0].tif',255 * output[0])

data = []
img = tiff.imread('crop/6120_2_2_2-3_crops.tif')
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)
data.append(img)
data = np.array(data).reshape((len(data), 3, ISZ, ISZ))
print(data.shape)
tiff.imsave('im_pred123.tif', data[0, :, :, :])


model = get_unet()
model.load_weights('unet_weight_ep25_unet_CHL.hdf5')
output = model.predict(data, batch_size=4, verbose=1)
print(type(output))
print('output.shape', output.shape)

tiff.imsave('mask_pred123.tif', output[0, :, :, :])
'''