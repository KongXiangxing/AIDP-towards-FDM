from turtle import shape
import numpy as np 
import pandas as pd 
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import joblib
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import SimpleITK as sitk
import cv2



def pkl(path):

    ct_path = os.path.join(path,'CT_low')
    pet_path = os.path.join(path,'PET')
    slices_ct =load_scan(ct_path)
    slices_pet = load_scan(pet_path)
    img_ct = get_pixels_hu(slices_ct)
    img_pet = get_pixels_hu(slices_pet)
    spacing_ct = np.array([slices_ct[0].SliceThickness, slices_ct[0].PixelSpacing[0], slices_ct[0].PixelSpacing[1]], dtype=np.float32)
    spacing_pet = np.array([slices_pet[0].SliceThickness, slices_pet[0].PixelSpacing[0], slices_pet[0].PixelSpacing[1]], dtype=np.float32)
    spacing = np.array([2.94227, 2.5631666, 2.5631666])
    resampled_ct, __ = resample_spacing(img_ct, spacing_ct, spacing)
    resampled_pet, __ = resample_spacing(img_pet, spacing_pet, spacing)
    c1 = abs(int((resampled_pet.shape[0]-resampled_ct.shape[0])/2))
    c2 = abs(int((resampled_pet.shape[0]+resampled_ct.shape[0])/2))
    h1 = abs(int((resampled_pet.shape[1]-resampled_ct.shape[1])/2))
    h2 = abs(int((resampled_pet.shape[1]+resampled_ct.shape[1])/2))
    w1 = abs(int((resampled_pet.shape[2]-resampled_ct.shape[2])/2))
    w2 = abs(int((resampled_pet.shape[2]+resampled_ct.shape[2])/2))
    if img_pet.shape[2] == 200:
        pet = resampled_pet[c1:c2,h1:h2,w1:w2]
        ct = resampled_ct
    elif img_pet.shape[2] == 144:
        ct = resampled_ct[c1:c2,h1:h2,w1:w2]
        pet = resampled_pet
    else:
        print('Shape is neither 144 nor 200.')
    data = np.array([ct,pet])
    joblib.dump(data,os.path.join(path,'data.pkl'))
    
    
    
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_spacing = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_spacing = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_spacing

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample_spacing(image, spacing, new_spacing):
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def resample(image, new_shape):
    real_resize_factor = new_shape / image.shape
    # image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
def segment_lung_mask(image,thr):

    binary_image = np.array(image > thr, dtype=np.int8)+1
    labels = measure.label(binary_image)

    background_label = labels[0,0,0]

    binary_image[background_label == labels] = 2
    binary_image -= 1 
    binary_image = 1-binary_image 

    # labels = measure.label(binary_image, background=0)
    # l_max = largest_label_volume(labels, bg=0)
    # if l_max is not None: 
    #     binary_image[labels != l_max] = 0

    return binary_image


def connected_domain_2(image, mask=True):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    img = image.copy()
    index_min = img <-125
    index_max = img>100
    img[index_min] = -1024
    img[index_max] = 1024
    _input = sitk.GetImageFromArray(img.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)

    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask

    if mask:
        output = (output > 0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    for i in range(output.shape[0]):
        output[i] = binary_fill_holes(output[i])
    #     contours, __ = cv2.findContours(output[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #     cv2.fillPoly(output[i],[contours[0]],1)
    return output

def readdcm(filepath,out_path):
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
    series_reader = sitk.ImageSeriesReader() 
    series_reader.SetFileNames(series_file_names) 
    images = series_reader.Execute()
    sitk.WriteImage(images, out_path)#保存为nii
    return images



def main_pre(file_path,shape,window_level=-600,window_width = 1600):
    data = joblib.load(file_path)
    ct = data[0]
    pet = data[1]
    num_layer = ct.shape[0]
    if num_layer>200:
        ct = ct[num_layer-200:num_layer-50,:,:]
        pet = pet[num_layer-200:num_layer-50,:,:]
    # ct_c = ct.copy()
    # index_bed1 = ct > 50 +0.5*350
    # index_bed2 = 50 < 0.5*350
    # ct_c[index_bed1] = -1024
    # ct_c[index_bed2] = 1024
    # print(ct.shape,ct_c.shape)
    human_mask = connected_domain_2(ct)
    window_max = window_level + 0.5*window_width
    window_min = window_level - 0.5*window_width
    index_min = ct <window_min
    index_max = ct>window_max
    ct[index_min] = -1024
    ct[index_max] = 1024
    ct[human_mask == 0] = -1024
    # pet[human_mask == 0] = 0
    nonzero_mask = np.zeros(ct.shape[1:], dtype=bool)
    for c in range(ct.shape[0]):
        this_mask = ct[c] >=-800
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    mask_voxel_coords = np.where(nonzero_mask != False)

    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1

    bbox = [[minzidx, maxzidx], [minxidx, maxxidx]]
    resizer = ( slice(bbox[0][0], bbox[0][1]),slice(bbox[1][0], bbox[1][1]))
    cropped_data_ct = []
    cropped_data_pet = []
    for c in range(ct.shape[0]):
        cropped_ct = ct[c][resizer]
        cropped_pet = pet[c][resizer]
        cropped_data_ct.append(cropped_ct[None])
        cropped_data_pet.append(cropped_pet[None])
    img_ct = np.vstack(cropped_data_ct)
    img_pet = np.vstack(cropped_data_pet)
    p1 = []
    p2 = []
    for i in range(3):
        if img_ct.shape[i] < shape[i]:
            a = int((shape[i]-img_ct.shape[i])/2)
            p1.append(a)
            p2.append(shape[i] - img_ct.shape[i] - a)
        else:
            p1.append(0)
            p2.append(0)
    img_ct = np.pad(img_ct,pad_width=((p1[0], p2[0]),
                                 (p1[1], p2[1]),
                                 (p1[2],p2[2])),mode="constant",constant_values=(-1024,-1024))
    img_pet = np.pad(img_pet,pad_width=((p1[0], p2[0]),
                                 (p1[1], p2[1]),
                                 (p1[2],p2[2])),mode="constant",constant_values=(0,0))
    c1 = abs(int((img_ct.shape[0]-shape[0])/2))
    c2 = abs(int((img_ct.shape[0]+shape[0])/2))
    h1 = abs(int((img_ct.shape[1]-shape[1])/2))
    h2 = abs(int((img_ct.shape[1]+shape[1])/2))
    w1 = abs(int((img_ct.shape[2]-shape[2])/2))
    w2 = abs(int((img_ct.shape[2]+shape[2])/2))
    img_ct = img_ct[c1:c2,h1:h2,w1:w2]
    img_pet = img_pet[c1:c2,h1:h2,w1:w2]
    # img_ct = np.array([img_ct])
    # img_pet = np.array([img_pet])
    # img_ct = img_ct +1000
    # img_ct[img_ct < 0] = 0
    # img_ct = img_ct + 1
    # img_ct = np.log(img_ct)
    return img_ct,img_pet

def bed(file_path,shape):
    data = joblib.load(file_path)
    ct = data[0]
    pet = data[1]
    num_layer = ct.shape[0]
    if num_layer>200:
        ct = ct[num_layer-200:num_layer-50,:,:]
        pet = pet[num_layer-200:num_layer-50,:,:]
# print(ct.shape,pet.shape)
    # human_mask = connected_domain_2(ct)
    # ct[human_mask == 0] = -1024
    # nonzero_mask = np.zeros(ct.shape[1:], dtype=bool)
    # for c in range(ct.shape[0]):
    #     this_mask = ct[c] >=-800
    #     nonzero_mask = nonzero_mask | this_mask
    # nonzero_mask = binary_fill_holes(nonzero_mask)
    # mask_voxel_coords = np.where(nonzero_mask != False)

    # minzidx = int(np.min(mask_voxel_coords[0]))
    # maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    # minxidx = int(np.min(mask_voxel_coords[1]))
    # maxxidx = int(np.max(mask_voxel_coords[1])) + 1

    # bbox = [[minzidx, maxzidx], [minxidx, maxxidx]]
    # resizer = ( slice(bbox[0][0], bbox[0][1]),slice(bbox[1][0], bbox[1][1]))
    # cropped_data_ct = []
    # cropped_data_pet = []
    # for c in range(ct.shape[0]):
    #     cropped_ct = ct[c][resizer]
    #     cropped_pet = pet[c][resizer]
    #     cropped_data_ct.append(cropped_ct[None])
    #     cropped_data_pet.append(cropped_pet[None])
    # img_ct = np.vstack(cropped_data_ct)
    # img_pet = np.vstack(cropped_data_pet)
    p1 = []
    p2 = []
    for i in range(3):
        if ct.shape[i] < shape[i]:
            a = int((shape[i]-ct.shape[i])/2)
            p1.append(a)
            p2.append(shape[i] - ct.shape[i] - a)
        else:
            p1.append(0)
            p2.append(0)
    img_ct = np.pad(ct,pad_width=((p1[0], p2[0]),
                                 (p1[1], p2[1]),
                                 (p1[2],p2[2])),mode="constant",constant_values=(-1024,-1024))
    img_pet = np.pad(pet,pad_width=((p1[0], p2[0]),
                                 (p1[1], p2[1]),
                                 (p1[2],p2[2])),mode="constant",constant_values=(0,0))
    c1 = abs(int((img_ct.shape[0]-shape[0])/2))
    c2 = abs(int((img_ct.shape[0]+shape[0])/2))
    h1 = abs(int((img_ct.shape[1]-shape[1])/2))
    h2 = abs(int((img_ct.shape[1]+shape[1])/2))
    w1 = abs(int((img_ct.shape[2]-shape[2])/2))
    w2 = abs(int((img_ct.shape[2]+shape[2])/2))
    img_ct = img_ct[c1:c2,h1:h2,w1:w2]
    img_pet = img_pet[c1:c2,h1:h2,w1:w2]
    # img_ct = np.array([img_ct])
    # img_pet = np.array([img_pet])
    return img_ct,img_pet
