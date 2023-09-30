import face_recognition as fc

import urllib.request
# import requests
import numpy as np
import pandas as pd
import os, time
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import base64, io, re

from face_encoding import encode_face_image

# a1_image = fc.load_image_file("input/a1.jpg")
# unknown_image = fc.load_image_file("input/b1.jpg")

def resize_image_with_aspect_ratio(input_path ):
    new_width = 512
    # Open the image
    image = Image.open(input_path)
    image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)
    # Calculate the new height while maintaining the aspect ratio
    aspect_ratio = image.width / image.height
    new_height = int(new_width / aspect_ratio)
    # Resize the image
    #print(new_width, new_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    #print(resized_image.width, resized_image.height)
    # Save the resized image
    resized_image.save(input_path)
    image.close()
    resized_image.close()


def cosine_similarity(x, y):
    # Ensure length of x and y are the same
    if len(x) != len(y) :
        return None
    # Compute the dot product between x and y
    dot_product = np.dot(x, y)
    # Compute the L2 norms (magnitudes) of x and y
    magnitude_x = np.sqrt(np.sum(x**2)) 
    magnitude_y = np.sqrt(np.sum(y**2))
    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    return cosine_similarity


def face_recg(path1, path2):
    img1 = fc.load_image_file(path1)
    img2 = fc.load_image_file(path2)

    try:
        face1_encoding = fc.face_encodings(img1)[0]
        face2_encoding = fc.face_encodings(img2)[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()

    known_faces = [
        face1_encoding
    ]

    results = fc.compare_faces(known_faces, face2_encoding)#, tolerance=0.9)
    result_d = fc.face_distance(known_faces, face2_encoding)
    return results, result_d

# start_time1 = time.time()
# path1 = r"G:\Face Recognition with Real-Time\FaceRecognitionTestData\vk11.PNG"
# resize_image_with_aspect_ratio(path1 )
# path2 = r"G:\Face Recognition with Real-Time\FaceRecognitionTestData\vk.PNG"
# resize_image_with_aspect_ratio(path2 )
# results, result_d = face_recg(path1, path2)
# print(results)
# print(result_d)
# req_time1 = round((time.time() - start_time1), 2)
# print("--- Time for FaceRecognition_main:  %s seconds ---" % (req_time1))

def crop_face(path ):
    try:
        img = fc.load_image_file(path,)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = fc.face_locations(img)  # top, right, bottom, left
        print(face_locations)
        top, right, bottom, left = face_locations[0]
        crop_img = img[top:bottom, left:right]
        cv2.imwrite(path, crop_img)
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    pass

def encode_face_image_old(path):
    # crop_face(path )
    # print("In encode_face_image func")
    img = fc.load_image_file( path   )
    face_encoding = []
    try:
        face_encoding = fc.face_encodings(img)
        # enc1 =  face1_encoding.tobytes()
        # enc1 =  face1_encoding.tolist()
        #print(type(enc1))
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    return face_encoding


def face_recog(enc_list, enc2):
    # fc.compare_faces(known_faces, face2_encoding)#, tolerance=0.9)
    result_fg = fc.compare_faces(enc_list, enc2, tolerance=0.35 )  # tolerance=0.6,
    flag = result_fg[0]
    # print( flag )
    
    result_d = fc.face_distance(enc_list, enc2)
    act_dist = round( result_d[0], 2)
    print("Actual Distance: ",act_dist)
    print("Note: Actual distance less than 0.35 then similar Faces")
    match_score = round( 1-result_d[0], 2)
    return flag, match_score
    
def score_convert(score, threshold = 0.65):
# def score_convert(score, threshold = 0.60):
    print("In score_convert Func score:",score)
    result = 90+int(score*10) if score >= threshold else 0+int(score*10)
    # result = 90+int(score*10)
    return result


def FaceRecognition_main(path1, known_faces, member_id):
    face_encoding_list = encode_face_image(path1, member_id)
    # face_encoding_list = encode_face_image(path1)   # old 
    # if face_encoding.size > 0:
    ret_val = [len(face_encoding_list), False, 0.01]

    if len(face_encoding_list)==1:
        list_enc = face_encoding_list      # list of encoding
        single_enc = known_faces[0]        # single encoding vector
        # roll_count = 13
        # single_enc0 = np.roll(single_enc, roll_count)
        # single_enc = np.roll(single_enc0, -roll_count)

        flag, match_score = face_recog(list_enc, single_enc )
        percentage_score = score_convert( match_score )
        percentage_score = percentage_score/100
        # print( flag, match_score )
        if flag:
            print("Face Match....")
            # func to convert fr_score to readable
            print("Converted Score:", percentage_score)
            ret_val[1] = True
            ret_val[2] = percentage_score
        else:
            print("Face Not match....")
            print("Converted Score:", percentage_score)
            ret_val[1] = False
            ret_val[2] = percentage_score 

        # face_encoding = face_encoding_list[0]
        # # known_faces : List required
        # result_d = fc.face_distance(known_faces, face_encoding)[0]
        # best_score = 1-result_d  # convert fc score
        # print("Face Recognition Result:", best_score)
        # result_d_agr_min = np.argmin(result_d)
        # best_score = result_d[result_d_agr_min]
        # known_face = known_faces[0]
        # result_d = cosine_similarity(known_face, face_encoding)
        # print("cosine similarity: ",result_d)

        # if best_score >= 0.65:     # Face found, Face matched, match score
        #     ret_val[1] = True
        #     ret_val[2] = best_score
        # else:                   # Face found, Not Face matched, other not used
        #     ret_val[1] = False
        #     ret_val[2] = best_score    
    return ret_val

def download_Image(file_url):
    print("--- Downloading Image File...")
    try:
        filename = Path(file_url).name
        file_path = os.path.join("Uploads", filename)
        #print(file_url, file_path)
        urllib.request.urlretrieve(file_url, file_path)
        resize_image_with_aspect_ratio(file_path)
        file_path = file_path.replace('\\', '/')
    except Exception as exception:
        print(exception)
        file_path = None
    finally:
        return file_path
    

def write_excel(path, new_dict):
    df = pd.read_excel(path)
    new_df = pd.DataFrame.from_dict(new_dict, orient='index').T
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_excel(path, index=False)
    print("Master Data saved....")
    print("Shape:", df.shape)
    return True
# url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Elon_Musk_Colorado_2022_%28cropped2%29.jpg/330px-Elon_Musk_Colorado_2022_%28cropped2%29.jpg"
# download_Image(url1)




class Base64_Image:

    def __init__(self, base64_img):
        self.base64_img = base64_img
        
    def find_extension(self):
        sent = self.base64_img
        pat = r'^data:image/([A-Za-z]+);base64'
        extension = re.match(pat, sent).group(1)
        return extension

    def get_encoded_str(self):
        sent = self.base64_img
        key = 'base64,'
        add_len = len(key)
        sent1 = sent[sent.find(key)+add_len:]
        return sent1
    
    def base64_to_image_save(self, base64_image, output_file_path):
        # Decode the base64 image string.
        image_data = base64.b64decode(base64_image)
        # Create an Image object from the decoded image data.
        image = Image.open(io.BytesIO(image_data))
        # Save the image to the filesystem.
        image.save(output_file_path)
        print(f"Image File saved successfully at {output_file_path}")
        
    def convert(self):
        output_file_path = None
        test_en = self.base64_img
        ext  = self.find_extension()
        ts = time.strftime('%d%m%y_%H%M%S', time.localtime())
        filename = f"img_{ts}.{ext}"
        output_file_path = os.path.join("Uploads", filename)
        
        encoded_str = self.get_encoded_str()
        self.base64_to_image_save(encoded_str, output_file_path)
        return output_file_path