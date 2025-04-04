import cv2
import numpy as np
import pydicom
import os
import SimpleITK as sitk
import joblib
from shapely import geometry as geo
from shapely import wkt 
from shapely import ops
from shapely.geometry import Polygon
from shapely.ops import nearest_points



class auto_extract(object):
    def extract_all(dcm_dir,CT_M_path,lung_path,SUV_path,CT_path,PET_M_path):
        data = sitk.ReadImage(CT_M_path)
        data_lung = joblib.load(lung_path)
        data_pet = sitk.ReadImage(SUV_path)
        data_ct = sitk.ReadImage(CT_path)
        data_pet_M = sitk.ReadImage(PET_M_path)
        lstFilesDCM = os.listdir(dcm_dir)
        mask = sitk.GetArrayFromImage(data)
        ROI = []
        for i in range(mask.shape[0]):
            n = np.sum(mask[i] == 1)
            ROI.append(n)
        index_max = ROI.index(np.max(ROI))
        mask_max = mask[index_max].astype(np.uint8)
        RefDs = pydicom.read_file(os.path.join(dcm_dir, lstFilesDCM[index_max]))
        spacing = RefDs.PixelSpacing[0]


        
        contours_tumor, hierarchy = cv2.findContours(mask_max,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
        if len(contours_tumor) != 0:
            points = np.array(contours_tumor[0], dtype=np.float32)
            ellipse_tumor = cv2.fitEllipse(points)
            major_axis_tumor = max(ellipse_tumor[1]) * spacing
            minor_axis_tumor = min(ellipse_tumor[1]) * spacing


            polygon = Polygon(contours_tumor[0].reshape(-1, 2))
            lung_local = data_lung[index_max].astype(np.uint8)
            lung_local[mask_max == 0] = 0
            if len(np.unique(lung_local)) == 1:
                L_R = "NaN"
                left_right = 170
                left_right_zero = 255
            else:
                left_right = np.unique(lung_local)[1]
                if left_right == 170:
                    left_right_zero = 255
                    L_R = 0
                else: 
                    left_right_zero = 170
                    L_R = 1
            

            data_lung[data_lung == left_right] = 1
            data_lung[data_lung == left_right_zero] = 0
            lung_max = data_lung[index_max].astype(np.uint8)
            contours_lung, __ = cv2.findContours(lung_max,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
            if len(contours_lung)==0:
                shortest_distance = 'NaN'
            else:
                polygon_lung = Polygon(contours_lung[0].reshape(-1, 2))
                boundary1 = polygon.boundary
                boundary2 = polygon_lung.boundary

                nearest_points_poly1_poly2 = nearest_points(boundary1, boundary2)
                shortest_distance = nearest_points_poly1_poly2[0].distance(nearest_points_poly1_poly2[1])
                if shortest_distance > 0 :
                    shortest_distance = -shortest_distance * spacing
                else:
                    intersection_boundary = boundary1.intersection(boundary2)
                    overlap_length = intersection_boundary.length
                    shortest_distance = overlap_length * spacing
            

            ct = sitk.GetArrayFromImage(data_ct)
            ct_max = ct[index_max]
            window_level = 30
            window_width = 250
            window_max = window_level + 0.5*window_width
            window_min = window_level - 0.5*window_width
            index_min = ct_max < window_min
            index_max = ct_max > window_max
            ct_max[index_max] = 0
            ct_max[index_min] = 0
            ct_max[mask_max == 0] = 0
            ct_max[ct_max != 0] = 1
            ct_max = ct_max.astype(np.uint8)
            CTR = np.sum(ct_max == 1 )/np.sum(mask_max == 1 )
            # if np.sum(ct_max == 1 ) == 0:
            #     CTR = 0
            # else:
            #     contours_CTR, hierarchy = cv2.findContours(ct_max,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
            #     points_CTR = np.array(contours_CTR[0], dtype=np.float32)
            #     if len(points_CTR)<5:
            #         CTR = 0
            #     else:
            #         ellipse_CTR = cv2.fitEllipse(points_CTR)
            #         major_axis_CTR = max(ellipse_CTR[1])
            #         minor_axis_CTR = min(ellipse_CTR[1])
            #         # polygon_CTR = Polygon(contours_CTR[0].reshape(-1, 2))
            #         CTR = major_axis_CTR/major_axis_tumor
            #         if CTR >=2:
            #             CTR = 0

            # SUVmax
            data_pet = sitk.GetArrayFromImage(data_pet)
            data_pet_M = sitk.GetArrayFromImage(data_pet_M)
            # print(data_pet_M)
            data_pet[data_pet_M == 0] = 0
            SUVmax = np.max(data_pet)

            return major_axis_tumor, minor_axis_tumor, L_R, shortest_distance, CTR, SUVmax
        else:
            return 'NaN','NaN','NaN','NaN','NaN','NaN'