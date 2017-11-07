import os
import numpy as np
import tifffile as tiff

img_h = 128
img_w = 128
n_labels = 2
smooth = 1e-12

def label_map(labels):
    label_map = np.ones([n_labels, img_h, img_w])
    for r in range(img_h):
        for c in range(img_w):
            label_map[labels[r][c], r, c] = 0
    label_map = label_map.astype(np.uint8)
    return label_map

if __name__ == '__main__':

    path = 'builds/'
    l = os.listdir(path)
    l.sort()

    data = []
    label = []
    # n = 169
    print('start load')

    len_list_of_img = len(l)

    for i in range(0, len_list_of_img, 2):
        if(i%100 == 0):
            print(i, end=' ')
        img = tiff.imread(path + l[i])
        img = img.astype(np.float32)
        img = img / 255
        gt = tiff.imread(path + l[i+1])
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        data.append(img)
        t = label_map(gt)
        label.append(t)
    label = np.array(label).reshape((int(len_list_of_img/2), 2,  img_h, img_w)).astype(np.uint8)
    data = np.array(data).reshape((int(len_list_of_img/2), 3, img_h, img_w)).astype(np.float32)
    np.save('build_128_data.npy', data)
    np.save('build_128_label.npy', label)
    
    print('load complete')