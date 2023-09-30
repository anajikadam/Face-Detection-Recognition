import os, time
import numpy as np
import pandas as pd
import math

import cv2
import mediapipe

from PIL import Image

import face_recognition  as fr
from PIL import Image
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt

def load_image( path ):
    img_base = cv2.imread( path )
    img_proc = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)
    return img_proc

def getLandmarkCenter(img, landmarks, facial_area_obj):
    lmrk_routes = []
    for source_idx, target_idx in facial_area_obj:
        # print(source_idx, target_idx)
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
    
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        
        lmrk_routes.append(relative_source)
        lmrk_routes.append(relative_target)
    
        # cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
    lmrk_min_x_pt = min(lmrk_routes, key=lambda x: x[0])
    lmrk_max_x_pt = max(lmrk_routes, key=lambda x: x[0])
    lmrk_min_y_pt = min(lmrk_routes, key=lambda x: x[1])
    lmrk_max_y_pt = max(lmrk_routes, key=lambda x: x[1])
    
    center = (int((lmrk_min_x_pt[0]+lmrk_max_x_pt[0])/2.0), int((lmrk_min_y_pt[1]+lmrk_max_y_pt[1])/2.0))
    # cv2.circle(img, center, 2, (255, 255, 255), thickness = 2)
    return center

# ======================================================================================
# Initialize
 
mp_face_detection = mediapipe.solutions.face_detection
face_detector =  mp_face_detection.FaceDetection( min_detection_confidence = 0.6)

faceModule = mediapipe.solutions.face_mesh
face_mesh = faceModule.FaceMesh(static_image_mode=True)

# ======================================================================================

def find_imp_cord( img):
    fd_results = face_detector.process(img)
    right_eye_center, left_eye_center = 0,0
    if fd_results.detections:
        if len(fd_results.detections)==1:
            print("Selected Number of Faces:", len(fd_results.detections))

            results = face_mesh.process(img )
            if results.multi_face_landmarks is None:  # image 180degree 
                new_img = Image.fromarray(img)
                img = np.array(new_img.rotate(180))
                results = face_mesh.process(img )
                
            landmarks = results.multi_face_landmarks[0]
            
            facial_area_obj = faceModule.FACEMESH_RIGHT_EYE
            right_eye_center = getLandmarkCenter(img, landmarks, facial_area_obj)
            
            facial_area_obj = faceModule.FACEMESH_LEFT_EYE
            left_eye_center = getLandmarkCenter(img, landmarks, facial_area_obj)
            # return right_eye_center, left_eye_center
        else:
            print("more than one face")
    else:
        print("No Face Detection")
    return right_eye_center, left_eye_center, img

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def align_face(img_obj, right_eye_center, left_eye_center):
    right_eye, left_eye = left_eye_center, right_eye_center
    
    left_eye_x = left_eye[0]
    left_eye_y = left_eye[1]
    right_eye_x = right_eye[0]
    right_eye_y =right_eye[1]
    #----------------------
    #find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        # print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")

    #----------------------
    #find angle
    a = euclidean_distance(left_eye, point_3rd) # math.dist(left_eye, point_3rd)
    b = euclidean_distance(right_eye, point_3rd)
    c = euclidean_distance(right_eye, left_eye)
    
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    #print("cos(a) = ", cos_a)
    angle = np.arccos(cos_a)
    #print("angle: ", angle," in radian")
    
    angle = (angle * 180) / math.pi
    # print("angle: ", angle," in degree")
    
    if direction == -1:
        angle = 90 - angle

    # print("angle: ", angle," in degree")
    #--------------------
    #rotate image
    
    new_img = Image.fromarray(img_obj)
    new_img = np.array(new_img.rotate(direction * angle))
    return new_img


def crop_face( img ):
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process( img )
    if results.multi_face_landmarks is None:
        return False, img
      
    landmarks = results.multi_face_landmarks[0]
    abc = mp_face_mesh.FACEMESH_FACE_OVAL
    df = pd.DataFrame(list( abc ), columns = ["p1", "p2"])
    
    routes_idx = []
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]
    
    for i in range(0, df.shape[0]):
        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]
        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)
    
    routes = []
    for source_idx, target_idx in routes_idx:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
            
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        routes.append(relative_source)
        routes.append(relative_target)
    
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)
     
    out = np.zeros_like(img)
    out[mask] = img[mask]
    
    df1 = pd.DataFrame(routes, columns = ["p1", "p2"])
    x1,y1,x2,y2 = min(df1['p1']), min(df1['p2']), max(df1['p1']), max(df1['p2']) 
    x1,y1,x2,y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2) 
    # img1 = img[y1:y2, x1:x2]
    img1 = out[y1:y2, x1:x2]
    return True, img1

def resize_image( img ):
    # print('Original Dimensions : ',img.shape)
    # scale_percent = 60 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    width, height = 250, 250
    width, height = 512, 512
    dim = (width, height)
    # print(dim)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # print('Resized Dimensions : ',resized.shape)
    return resized

def encode_face_image( path, member_id ):
    # path = path.replace('\\','/')
    # print(path)
    
    face_encoding = []
    try:
        fileName = path.split('/')[-1]
        out_filename = f"fd_{member_id}_{time.strftime('%d%m%y%H%M%S', time.localtime())}_{fileName}"
        output_file_path = os.path.join('Temp', out_filename)
            
        start_time= time.time()

        img_obj = load_image( path )
        img_obj_aug = img_obj.copy()

        right_eye_center, left_eye_center, img_upd = find_imp_cord(img_obj_aug )

        if right_eye_center != 0:
            img_best = align_face(img_upd, right_eye_center, left_eye_center)
            flag, img_crop = crop_face( img_best )
            if flag:
                final_img = img_crop.copy()
                # resized crop image width, height = 250, 250 or 512, 512*
                final_img_resized = resize_image( final_img )
                Image.fromarray(final_img_resized).save(output_file_path)
                # save crop file
                image = fr.load_image_file(output_file_path)
                fc_landmark = fr.face_landmarks(image)
                # print(fc_landmark)
                if fc_landmark:
                    # Face Landmark found using Face Recognition
                    # face_encoding = fr.face_encodings(image)[0]
                    face_encoding = fr.face_encodings(image)
                    # print(fc_encoding)
                    req_time = round((time.time() - start_time), 2)
                else:
                    os.remove(output_file_path)
                    print("Please ensure Entire Face is visible.....!!!!!")
                    req_time = round((time.time() - start_time), 2)
        else:
            print("Abort...")
            req_time = round((time.time() - start_time), 2)

        print("---  Face Encoding Overall Time in %s seconds ---" % (req_time))
    except Exception as ex:
        print(ex)
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    return face_encoding


# path = r"Img/yr1.jpg"
# encoding_list = encode_face_image( path )
# if len(encoding_list) == 1:
#     encoding = encoding_list[0]
#     # print(type(encoding))
#     # convert to str
#     encode_face_str = str( encoding.tolist() )
#     # print(type(encode_face_str))
#     print(encode_face_str)
# else:
#     print("Multiple Face Detected. OR No Face Detected.......!!!")


# ======================================================================================

# path = r"Img/test3.jpg"
# fileName = path.split('/')[-1]
# out_filename = f"Temp_{time.strftime('%d%m%y%H%M%S', time.localtime())}_{fileName}.jpg"
# output_file_path = os.path.join('Temp', out_filename)

# start_time= time.time()

# img_obj = load_image( path )
# img_obj_aug = img_obj.copy()

# right_eye_center, left_eye_center, img_upd = find_imp_cord(img_obj_aug )

# if right_eye_center != 0:
#     img_best = align_face(img_upd, right_eye_center, left_eye_center)
#     flag, img_crop = crop_face( img_best )
#     if flag:
#         final_img = img_crop.copy()
#         Image.fromarray(final_img).save(output_file_path)
#         # save crop file
#         image = fr.load_image_file(output_file_path)
#         fc_landmark = fr.face_landmarks(image)
#         # print(fc_landmark)
#         if fc_landmark:
#             # Face Landmark found using Face Recognition
#             fc_encoding = fr.face_encodings(image)[0]
#             # print(fc_encoding)
#             req_time = round((time.time() - start_time), 2)
#         else:
#             os.remove(output_file_path)
#             print("Please ensure Entire Face is visible.....!!!!!")
#             req_time = round((time.time() - start_time), 2)
# else:
#     print("Abort...")
#     req_time = round((time.time() - start_time), 2)

# print("---  Overall Time in %s seconds ---" % (req_time))
