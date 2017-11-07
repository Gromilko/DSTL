import tifffile as tiff
from skimage.transform import resize
import numpy as np
import csv
import shapely.wkt
import cv2

import sys

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

def get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

# Create a mask from polygons:

def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def show_mask(m):
    # hack for nice display
    tiff.imshow(255 * np.stack([m, m, m]))
    #tiff.imsave('6120_2_2_pred_mask_bin26.tif', 255 * np.stack([m, m, m]))

def label_map(labels):
    #print('----------')
    #print(labels.shape)
    label_map = np.zeros([3200, 3200, 2])
    for r in range(3200):
        for c in range(3200):
            label_map[r, c, labels[r][c]] = 1
    #print(label_map.shape)
    #print('----------')
    return label_map



IMAGE_SIZE = 3328
#IM_ID = '6120_2_2'

#trees_ids_list = ['6010_1_2', '6010_4_2', '6010_4_4', '6040_1_0', '6040_1_3',
#               '6040_2_2', '6040_4_4', '6060_2_3', '6070_2_3', '6090_2_0',
#               '6100_1_3', '6100_2_2', '6100_2_3', '6110_1_2', '6110_3_1',
#               '6110_4_0', '6120_2_0', '6120_2_2', '6140_1_2', '6140_3_1',
#               '6150_2_3', '6160_2_1', '6170_0_4', '6170_2_4', '6170_4_1']

im_ids_list = ['6060_2_3', '6070_2_3', '6100_1_3', '6100_2_2', '6100_2_3', '6110_1_2',
               '6110_3_1', '6110_4_0', '6120_2_0', '6120_2_2', '6140_1_2', '6140_3_1']

# im_ids_list = ['6070_2_3']

POLY_TYPE = '1'

# ищем самые большие габариты у картинок
'''
img_max_size = [0, 0]
for i, IM_ID in enumerate(im_ids_list):
    im_rgb = tiff.imread('three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
    im_size = im_rgb.shape[:2]
    print(i, IM_ID, im_size)
    img_max_size[0] = max(im_size[0], img_max_size[0])
    img_max_size[1] = max(im_size[1], img_max_size[1])
print(img_max_size)
'''
for count, IM_ID in enumerate(im_ids_list):
    print(count, IM_ID)
    im_rgb = tiff.imread('../data/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
    # a = resize(im_rgb, (3350, 3403, 3), preserve_range=True)
    #im_rgb = np.array(im_rgb, dtype=np.float32)
    im_size = im_rgb.shape[:2]
    # TODO: хранить размеры в списке

    x_max = y_min = None
    for _im_id, _x, _y in csv.reader(open('../data/grid_sizes.csv')):
        if _im_id == IM_ID:
            x_max, y_min = float(_x), float(_y)
            break

    x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)

    train_polygons = None
    for _im_id, _poly_type, _poly in csv.reader(open('../data/train_wkt_v4.csv')):
        if _im_id == IM_ID and _poly_type == POLY_TYPE:
            train_polygons = shapely.wkt.loads(_poly)
            break
    # print("{} = pol is empty: {}".format(IM_ID, train_polygons.is_empty))
    train_polygons_scaled = shapely.affinity.scale(
        train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    train_mask = mask_for_polygons(train_polygons_scaled)
    im = resize(im_rgb, (IMAGE_SIZE, IMAGE_SIZE, 3), preserve_range=True, mode='reflect').astype(np.uint16)

    im = np.array(im, dtype=np.uint16)
    train_mask = mask_for_polygons(train_polygons_scaled)
    train_mask = np.array(train_mask, dtype=np.uint8)

    #tiff.imsave('train_mask.tif', train_mask)
    ms = resize(train_mask, (IMAGE_SIZE, IMAGE_SIZE), preserve_range=True, mode='reflect').astype(np.uint8)
    ms = np.array(ms, dtype=np.uint8)
    #tiff.imsave('6060_crops.tif', im)
    #tiff.imsave('6060_crops_mask.tif', ms.astype(np.uint16))

    for i, y in enumerate(range(0, IMAGE_SIZE, 128)):
        for j, x in enumerate(range(0, IMAGE_SIZE, 128)):
            tiff.imsave('builds/{}_{}-{}_crops.tif'.format(IM_ID, i, j), im[x:x+128, y:y+128, :])
            tiff.imsave('builds/{}_{}-{}_crops_mask.tif'.format(IM_ID, i, j), ms[x:x+128, y:y+128])