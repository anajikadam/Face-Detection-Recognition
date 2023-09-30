import uvicorn
from app_req import UVI_APP_PORT, UVI_APP_HOST
from pydantic import BaseModel

import time
import json
from typing import List, Optional, Union
import shutil
from fastapi import FastAPI, Form, Query, File, UploadFile, Response, status, Body
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

import pandas as pd
import numpy as np
import pickle
import re, os
from pathlib import Path
import string

from app_helper import face_recg
# from app_helper import face_recg, encode_face_image
from app_helper import download_Image, FaceRecognition_main, Base64_Image
from app_helper import write_excel, resize_image_with_aspect_ratio

from face_encoding import encode_face_image

from app_req import UVI_APP_HOST, UVI_APP_PORT

FastAPI_app = FastAPI()

# FastAPI_app.mount("/static", StaticFiles(directory="static"), name="static")
# FastAPI_app.mount("/Output", StaticFiles(directory="Output"), name="Output")
FastAPI_app.mount("/Uploads", StaticFiles(directory="Uploads"), name="Uploads")
templates = Jinja2Templates(directory="templates")


FastAPI_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(HTTPSRedirectMiddleware)
# @FastAPI_app.on_event('startup')
# async def load_model():
#      # if this file is run on its own - perform downloads
#     # initial_downloads()
#     global var1
#     print("Initial vars loaded...!")

@FastAPI_app.get("/")
@FastAPI_app.get("/api/")
def read_root(request: Request, ):
    print("--------/api/ In url read_root  --------") 
    data = {"Message-------------------------------": f"Welcome, Face Recognition API app",
            "Docs URL------------------------------": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/docs",
            "API RegisterMemberFace with URL-------": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/api/RegisterMemberFace/",
            "API Login with URL--------------------": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/api/checkMemberFaceRecognition/",
            "API RegisterMemberFace with Base64----": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/api/v2/RegisterMemberFace/",
            "API Login with Base64-----------------": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/api/v2/checkMemberFaceRecognition/",
            "UAT API RegisterMemberFace with File--": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/UAT/api/RegisterMemberFace/",
            "UAT API Login with File---------------": f"http://{UVI_APP_HOST}:{UVI_APP_PORT}/UAT/api/checkMemberFaceRecognition/"
            }
    ret_dict = {
                "data": data,
                "status":200,
                "message": "Success"
                }
    return ret_dict

class RegisterMemberFaceInput(BaseModel):
    member_id: int = 111
    face_url: str = "https://hsm.care-tpa.com/CareTPA_FaceDetection/22082301.PNG"

@FastAPI_app.post("/api/RegisterMemberFace/",)
# async def get_RegisterMemberFace(request: Request, ID: int = Form(...), file1: UploadFile = File(...),):
async def get_RegisterMemberFace(request: Request, item: RegisterMemberFaceInput):
    timestamp01 = time.strftime('%d-%m-%y_%H:%M:%S', time.localtime())

    start_time0 = time.time()
    print(f"-------- In url get_RegisterMemberFace func --------{timestamp01}")
    member_id = item.member_id
    img_file_url = item.face_url

    start_time1 = time.time()
    img_file_path = download_Image(img_file_url)
    req_time1 = round((time.time() - start_time1), 2)
    print("--- Time for Downloading Image File:  %s seconds ---" % (req_time1))

    if img_file_path:
        start_time2 = time.time()
        encoding_list = encode_face_image( img_file_path, member_id )

        req_time2 = round((time.time() - start_time2), 2)
                
        if len(encoding_list) == 1:
            encoding = encoding_list[0]
            data = {
            "member_id":member_id,
            "image_encoding": str( encoding.tolist() )
            }
            ret_dict = {
                    "data": data,
                    "status":200,
                    "message": "Success"
                    }
        else:
            ret_dict = {
                    "data": {},
                    "status":400,
                    "message": "Multiple Face Detected. OR No Face Detected.......!!!"
                    }
    else:
        ret_dict = {
                    "data": {},
                    "status":401,
                    "message": "Error in Image URL- Not Downloadable"
                    }
    req_time0 = round((time.time() - start_time0), 2)
    print("--- Time for /api/RegisterMemberFace:  %s seconds ---" % (req_time0))
    master_data = {
            'TimeStamp':timestamp01,
            'member_id': member_id,
            'Image_URL':img_file_url,
            'faces_found': len(encoding_list),
            'request_time':req_time0,
            'download_time':req_time1,
            'face_detect_time':req_time2
    }
    path = "LogMaster/FaceRegister-Master.xlsx"
    write_excel(path, master_data)
    # os.remove(img_file_path)
    return ret_dict

class MemberFaceRecognitionInput(BaseModel):
    member_id: int = 1234
    policy_id: int = 2345
    test_image_url: str = "https://hsm.care-tpa.com/CareTPA_FaceDetection/22082301.PNG"
    member_face_encoding: str = "[-0.0909091904759407, 0.1089334711432457, 0.06684223562479019, -0.07395923882722855, -0.03583826869726181, -0.0314648263156414, -0.040294349193573, -0.11419316381216049, 0.136981800198555, -0.12103697657585144, 0.2578197717666626, -0.0024862950667738914, -0.23648595809936523, -0.144810289144516, -0.05710684508085251, 0.07069094479084015, -0.1634160727262497, -0.11451488733291626, -0.10289400815963745, -0.0541234165430069, -0.032573096454143524, -0.018703417852520943, 0.06589458882808685, -0.061664797365665436, -0.10237924754619598, -0.3321395814418793, -0.11357129365205765, -0.05899639427661896, 0.06700042635202408, -0.05167164281010628, -0.03655770421028137, 0.028977787122130394, -0.23405314981937408, -0.08891083300113678, 0.010111266747117043, 0.027953067794442177, 0.05193885415792465, 0.0743478313088417, 0.1648721843957901, 0.04011787474155426, -0.12023399025201797, 5.2123330533504486e-05, 0.09183575212955475, 0.2385377585887909, 0.1859736442565918, 0.10500796139240265, -0.023983469232916832, -0.07553423941135406, 0.0767420157790184, -0.21393051743507385, 0.11395397782325745, 0.12128400057554245, 0.2002648264169693, 0.019568338990211487, 0.1421736478805542, -0.1395939439535141, 0.01734999567270279, 0.07192680239677429, -0.184594064950943, 0.06755614280700684, 0.07933638244867325, 0.019575532525777817, -0.04672486335039139, -0.006910089403390884, 0.24094466865062714, 0.1022283062338829, -0.1216793805360794, -0.08404751121997833, 0.1819698065519333, -0.1371476650238037, 0.049793541431427, 0.06854862719774246, -0.024226268753409386, -0.12485931068658829, -0.23512054979801178, 0.03619860112667084, 0.3841882348060608, 0.18096446990966797, -0.15679119527339935, 0.011309189721941948, -0.13344590365886688, -0.09811724722385406, 0.04023737460374832, -0.027798041701316833, -0.19265711307525635, 0.07264231890439987, -0.05085498094558716, 0.028025353327393532, 0.14156989753246307, 0.06258472800254822, -0.012615400366485119, 0.23029005527496338, -0.01528950035572052, 0.04527746140956879, 0.05780274420976639, -0.02712545171380043, -0.10124830156564713, -0.048038776963949203, -0.1552034169435501, -0.0022468101233243942, -0.017349570989608765, -0.1666831225156784, 0.006396395154297352, 0.05082995072007179, -0.1727258563041687, 0.07696978747844696, -0.007979406975209713, -0.042234502732753754, -0.03256035968661308, 0.03974447771906853, -0.06547007709741592, -0.03428993001580238, 0.11372429877519608, -0.3139425218105316, 0.19240130484104156, 0.13260143995285034, -0.0010428528767079115, 0.12659595906734467, 0.10480388253927231, 0.05258183181285858, 0.019175684079527855, -0.040780939161777496, -0.037633106112480164, -0.12751798331737518, 0.003597597125917673, -0.1328555792570114, 0.14249061048030853, 0.03641875088214874]"

@FastAPI_app.post("/api/checkMemberFaceRecognition/",)
async def FaceRecognition(request: Request, item: MemberFaceRecognitionInput):
    timestamp01 = time.strftime('%d-%m-%y_%H:%M:%S', time.localtime())
    start_time0 = time.time()

    def str_To_list(sent1):
        return np.array([float(i.strip()) for i in sent1.replace('[', '').replace(']','').split(',')])

    print(f"-------- In url FaceRecognition func --------{timestamp01}")
    member_id = item.member_id
    policy_id = item.policy_id
    test_img_url = item.test_image_url
    enc_data = item.member_face_encoding

    start_time1 = time.time()
    test_img_path = download_Image(test_img_url)
    
    if test_img_path is None:
        ret_dict = {
                    "data": {},
                    "status":400,
                    "message": "Error in Image URL- Not Downloadable"
                    }
        return ret_dict

    req_time1 = round((time.time() - start_time1), 2)
    print("--- Time for Downloading Image File:  %s seconds ---" % (req_time1))

    start_time2 = time.time()
    known_imgs_enc = [str_To_list(enc_data)]
    num_faces, matched_flag, best_score = FaceRecognition_main(test_img_path, known_imgs_enc, member_id)
    req_time2 = round((time.time() - start_time2), 2)
    print("--- Time for FaceRecognition_main:  %s seconds ---" % (req_time2))

    data = {
        'member_id': member_id,
        'policy_id': policy_id,
        'faces_found': num_faces,
        'face_matched':matched_flag,
        "matched_score": int( round(best_score, 2) *100 )
    }

    if num_faces == 0:
        ret_dict = {
                "data": data,
                "status":401,
                "message": "Face Not Detected."
                }        
    elif num_faces == 1:
        ret_dict = {
                    "data": data,
                    "status":200,
                    "message": "Success"
                    }
    else:
        ret_dict = {
                "data": data,
                "status":402,
                "message": "Multiple faces Detected."
                }
    req_time0 = round((time.time() - start_time0), 2)
    print("--- Time for /api/checkMemberFaceRecognition:  %s seconds ---" % (req_time0))
    # save all data for test
    master_data = {
            'TimeStamp':timestamp01,
            'member_id': member_id,
            'policy_id': policy_id,
            'TestImage_URL':test_img_url,
            'faces_found': num_faces,
            'face_matched':matched_flag,
            'matched_score': int( round(best_score, 2) *100 ),
            'request_time':req_time0,
            'download_time':req_time1,
            'face_recog_time':req_time2
        }
    master_path = "LogMaster/FaceRecognition_Login-Master.xlsx"
    write_excel(master_path, master_data)
    # os.remove(test_img_path)
    return ret_dict
# __________________________


class RegisterMemberFaceInput(BaseModel):
    member_id: int = 111
    base64_image: str = "Put your Base64 image encoding here"

@FastAPI_app.post("/api/v2/RegisterMemberFace/",)
async def get_RegisterMemberFace(request: Request, item: RegisterMemberFaceInput):
    timestamp01 = time.strftime('%d-%m-%y_%H:%M:%S', time.localtime())

    start_time0 = time.time()
    print(f"-------- In url v2 get_RegisterMemberFace func --------{timestamp01}")
    member_id = item.member_id
    encode_image = item.base64_image
    print("Len of Image Encoding", len(encode_image))
    start_time1 = time.time()
    img_file_path = None
    try:
        obj = Base64_Image(encode_image)
        img_file_path = obj.convert()
        print(img_file_path)
    except Exception as ex:
        print(ex)
    # img_file_path = download_Image(img_file_url)
    req_time1 = round((time.time() - start_time1), 2)
    print("--- Time for Decode Image File:  %s seconds ---" % (req_time1))

    if img_file_path:
        start_time2 = time.time()
        img_file_path = img_file_path.replace('\\','/')
        encoding_list = encode_face_image( img_file_path, member_id )

        req_time2 = round((time.time() - start_time2), 2)
                
        if len(encoding_list) == 1:
            encoding = encoding_list[0]
            data = {
            "member_id":member_id,
            "image_encoding": str( encoding.tolist() )
            }
            ret_dict = {
                    "data": data,
                    "status":200,
                    "message": "Success"
                    }
        else:
            ret_dict = {
                    "data": {},
                    "status":400,
                    "message": "Multiple Face Detected. OR No Face Detected.......!!!"
                    }
    else:
        ret_dict = {
                    "data": {},
                    "status":401,
                    "message": "Error in Image Encoding"
                    }
        return ret_dict
    req_time0 = round((time.time() - start_time0), 2)
    print("--- Time for /api/v2/RegisterMemberFace:  %s seconds ---" % (req_time0))
    master_data = {
            'TimeStamp':timestamp01,
            'member_id': member_id,
            'Image_URL':"Base64 Image Encoding",
            'faces_found': len(encoding_list),
            'request_time':req_time0,
            'download_time':req_time1,
            'face_detect_time':req_time2
    }
    path = "LogMaster/FaceRegister-Master.xlsx"
    write_excel(path, master_data)
    # os.remove(img_file_path)
    return ret_dict

class MemberFaceRecognitionInput(BaseModel):
    member_id: int = 1234
    policy_id: int = 2345
    base64_image: str = "Put your Base64 image encoding here"
    member_face_encoding: str = "[-0.0909091904759407, 0.1089334711432457, 0.06684223562479019, -0.07395923882722855, -0.03583826869726181, -0.0314648263156414, -0.040294349193573, -0.11419316381216049, 0.136981800198555, -0.12103697657585144, 0.2578197717666626, -0.0024862950667738914, -0.23648595809936523, -0.144810289144516, -0.05710684508085251, 0.07069094479084015, -0.1634160727262497, -0.11451488733291626, -0.10289400815963745, -0.0541234165430069, -0.032573096454143524, -0.018703417852520943, 0.06589458882808685, -0.061664797365665436, -0.10237924754619598, -0.3321395814418793, -0.11357129365205765, -0.05899639427661896, 0.06700042635202408, -0.05167164281010628, -0.03655770421028137, 0.028977787122130394, -0.23405314981937408, -0.08891083300113678, 0.010111266747117043, 0.027953067794442177, 0.05193885415792465, 0.0743478313088417, 0.1648721843957901, 0.04011787474155426, -0.12023399025201797, 5.2123330533504486e-05, 0.09183575212955475, 0.2385377585887909, 0.1859736442565918, 0.10500796139240265, -0.023983469232916832, -0.07553423941135406, 0.0767420157790184, -0.21393051743507385, 0.11395397782325745, 0.12128400057554245, 0.2002648264169693, 0.019568338990211487, 0.1421736478805542, -0.1395939439535141, 0.01734999567270279, 0.07192680239677429, -0.184594064950943, 0.06755614280700684, 0.07933638244867325, 0.019575532525777817, -0.04672486335039139, -0.006910089403390884, 0.24094466865062714, 0.1022283062338829, -0.1216793805360794, -0.08404751121997833, 0.1819698065519333, -0.1371476650238037, 0.049793541431427, 0.06854862719774246, -0.024226268753409386, -0.12485931068658829, -0.23512054979801178, 0.03619860112667084, 0.3841882348060608, 0.18096446990966797, -0.15679119527339935, 0.011309189721941948, -0.13344590365886688, -0.09811724722385406, 0.04023737460374832, -0.027798041701316833, -0.19265711307525635, 0.07264231890439987, -0.05085498094558716, 0.028025353327393532, 0.14156989753246307, 0.06258472800254822, -0.012615400366485119, 0.23029005527496338, -0.01528950035572052, 0.04527746140956879, 0.05780274420976639, -0.02712545171380043, -0.10124830156564713, -0.048038776963949203, -0.1552034169435501, -0.0022468101233243942, -0.017349570989608765, -0.1666831225156784, 0.006396395154297352, 0.05082995072007179, -0.1727258563041687, 0.07696978747844696, -0.007979406975209713, -0.042234502732753754, -0.03256035968661308, 0.03974447771906853, -0.06547007709741592, -0.03428993001580238, 0.11372429877519608, -0.3139425218105316, 0.19240130484104156, 0.13260143995285034, -0.0010428528767079115, 0.12659595906734467, 0.10480388253927231, 0.05258183181285858, 0.019175684079527855, -0.040780939161777496, -0.037633106112480164, -0.12751798331737518, 0.003597597125917673, -0.1328555792570114, 0.14249061048030853, 0.03641875088214874]"

@FastAPI_app.post("/api/v2/checkMemberFaceRecognition/",)
async def FaceRecognition(request: Request, item: MemberFaceRecognitionInput):
    timestamp01 = time.strftime('%d-%m-%y_%H:%M:%S', time.localtime())
    start_time0 = time.time()

    def str_To_list(sent1):
        return np.array([float(i.strip()) for i in sent1.replace('[', '').replace(']','').split(',')])

    print(f"-------- In url v2 FaceRecognition func --------{timestamp01}")
    member_id = item.member_id
    policy_id = item.policy_id
    enc_data = item.member_face_encoding
    encode_image = item.base64_image
    print("Len of Image Encoding", len(encode_image))
    start_time1 = time.time()
    test_img_path = None
    try:
        obj = Base64_Image(encode_image)
        test_img_path = obj.convert()
        print(test_img_path)
    except Exception as ex:
        print(ex)
    # test_img_path = download_Image(test_img_url)
    
    if test_img_path is None:
        ret_dict = {
                    "data": {},
                    "status":400,
                    "message": "Error in Image Encoding"
                    }
        return ret_dict

    req_time1 = round((time.time() - start_time1), 2)
    print("--- Time for Encoding Image File:  %s seconds ---" % (req_time1))

    start_time2 = time.time()
    known_imgs_enc = [str_To_list(enc_data)]
    test_img_path = test_img_path.replace('\\','/')
    num_faces, matched_flag, best_score = FaceRecognition_main(test_img_path, known_imgs_enc, member_id)
    req_time2 = round((time.time() - start_time2), 2)
    print("--- Time for v2 FaceRecognition_main:  %s seconds ---" % (req_time2))

    data = {
        'member_id': member_id,
        'policy_id': policy_id,
        'faces_found': num_faces,
        'face_matched':matched_flag,
        "matched_score": int( round(best_score, 2) *100 )
    }

    if num_faces == 0:
        ret_dict = {
                "data": data,
                "status":401,
                "message": "Face Not Detected."
                }        
    elif num_faces == 1:
        ret_dict = {
                    "data": data,
                    "status":200,
                    "message": "Success"
                    }
    else:
        ret_dict = {
                "data": data,
                "status":402,
                "message": "Multiple faces Detected."
                }
        return ret_dict
    req_time0 = round((time.time() - start_time0), 2)
    print("--- Time for /api/v2/checkMemberFaceRecognition:  %s seconds ---" % (req_time0))
    # save all data for test
    master_data = {
            'TimeStamp':timestamp01,
            'member_id': member_id,
            'policy_id': policy_id,
            'TestImage_URL':"Base64 Image Encoding",
            'faces_found': num_faces,
            'face_matched':matched_flag,
            'matched_score': int( round(best_score, 2) *100 ),
            'request_time':req_time0,
            'download_time':req_time1,
            'face_recog_time':req_time2
        }
    master_path = "LogMaster/FaceRecognition_Login-Master.xlsx"
    write_excel(master_path, master_data)
    # os.remove(test_img_path)
    return ret_dict
# __________________________

def write_excel_UAT(new_dict):
    path = "LogMaster/UAT_LocalFaceRegistry.xlsx"
    df = pd.read_excel(path)
    
    ids = new_dict['member_id']
    file_name = Path(new_dict['file_name']).name
    df_file_names = df['file_name'].apply(lambda x: Path(x).name)
    if ids in list(df['member_id']) or file_name in df_file_names:
        print("Data repeated, so not saved!")
        return False
    new_df = pd.DataFrame.from_dict(new_dict, orient='index').T
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_excel(path, index=False)
    print("Data saved...")
    print("Shape:", df.shape)
    return True

def get_member_face_encoding(id):
    path = "LogMaster/UAT_LocalFaceRegistry.xlsx"
    df = pd.read_excel(path)
    image_encoding = list(df[df['member_id']==id]['image_encoding'].values)[0]
    return image_encoding

@FastAPI_app.post("/UAT/api/RegisterMemberFace/",)
async def get_RegisterMemberFace(request: Request, ID: int = Form(...), files: UploadFile = File(...),):
    timestamp01 = time.strftime('%d-%m-%y_%H:%M:%S', time.localtime())

    start_time0 = time.time()
    print("-------- In url get_RegisterMemberFace func --------")
    member_id = ID
    t = time.localtime()
    img_file = files.filename
    # print(img_file_1, img_file_2)
    timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
    img_file_1 = f"{timestamp}_{img_file}"
    img_file_path = os.path.join("UAT-Uploads", img_file_1)

    with open( img_file_path, "wb") as buffer:
        shutil.copyfileobj(files.file, buffer)
    resize_image_with_aspect_ratio(img_file_path)
    # in windows path present '\', need to replace with '/'
    img_file_path = img_file_path.replace('\\','/')
    if img_file_path:
        start_time2 = time.time()
        encoding_list = encode_face_image(img_file_path, member_id)
        req_time2 = round((time.time() - start_time2), 2)
        if len(encoding_list) == 1:
            encoding = encoding_list[0]
            data = {
                    "member_id":member_id,
                    "image_encoding": str( encoding.tolist() )
                    }
            new_dict = {'member_id':member_id, 'image_encoding': data["image_encoding"],'file_name':img_file_path}
            flag = write_excel_UAT(new_dict)
            if flag:
                ret_dict = {
                            "data": data,
                            "status":200,
                            "message": "Success"
                            }
            else:
                ret_dict = {
                            "data": {},
                            "status":201,
                            "message": "Warning: Same Member ID found in DataBase with face encoding (Change Member ID and try again)"
                            } 
        else:
            ret_dict = {
                    "data": {},
                    "status":400,
                    "message": "Multiple Face Detected. OR No Face Detected.......!!!"
                    }
    else:
        ret_dict = {
                    "data": {},
                    "status":400,
                    "message": "Error in Test Image URL- Not Downloadable"
                    }
    req_time0 = round((time.time() - start_time0), 2)
    print("--- Time for /api/RegisterMemberFace:  %s seconds ---" % (req_time0))
    master_data = {
                'TimeStamp':timestamp01,
                'member_id': member_id,
                'Image_URL':img_file_path,
                'faces_found': len(encoding_list),
                'request_time':req_time0,
                'download_time':0,
                'face_detect_time':req_time2
            }
    path = "LogMaster/FaceRegister-Master.xlsx"
    write_excel(path, master_data)
    return ret_dict


@FastAPI_app.post("/UAT/api/checkMemberFaceRecognition/",)
async def FaceRecognition(request: Request, member_id: int = Form(...),
                                            policy_id: int = Form(...),
                                            #member_face_encoding: str = Form(...),
                                            file: UploadFile = File(...),):
    timestamp01 = time.strftime('%d-%m-%y_%H:%M:%S', time.localtime())
    start_time0 = time.time()

    def str_To_list(sent1):
        return np.array([float(i.strip()) for i in sent1.replace('[', '').replace(']','').split(',')])

    print("-------- In url FaceRecognition func --------")

    
    img_file = file.filename
    # print(img_file_1, img_file_2)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
    img_file_1 = f"Test_{timestamp}_{img_file}"
    test_img_path = os.path.join("UAT-Uploads", img_file_1)

    with open( test_img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resize_image_with_aspect_ratio(test_img_path)
    
    enc_data = get_member_face_encoding(member_id)
    known_imgs_enc = [str_To_list(enc_data)]
    start_time2 = time.time()
    # in windows path present '\', need to replace with '/'
    test_img_path = test_img_path.replace('\\','/')
    num_faces, matched_flag, best_score = FaceRecognition_main(test_img_path, known_imgs_enc, member_id)
    req_time2 = round((time.time() - start_time2), 2)
    print("--- Time for FaceRecognition_main:  %s seconds ---" % (req_time2))

    data = {
        'member_id': member_id,
        'policy_id': policy_id,
        'faces_found': num_faces,
        'face_matched':matched_flag,
        "matched_score": int( round(best_score, 2) *100 )
    }

    if num_faces == 0:
        ret_dict = {
                "data": data,
                "status":401,
                "message": "Face Not Detected."
                }        
    elif num_faces == 1:
        ret_dict = {
                    "data": data,
                    "status":200,
                    "message": "Success"
                    }
    else:
        ret_dict = {
                "data": data,
                "status":400,
                "message": "Multiple faces Detected."
                }
        
    req_time0 = round((time.time() - start_time0), 2)
    print("--- Time for /api/checkMemberFaceRecognition:  %s seconds ---" % (req_time0))
    master_data = {
            'TimeStamp':timestamp01,
            'member_id': member_id,
            'policy_id': policy_id,
            'TestImage_URL':test_img_path,
            'faces_found': num_faces,
            'face_matched':matched_flag,
            'matched_score': int( round(best_score, 2) *100 ),
            'request_time':req_time0,
            'download_time':0,
            'face_recog_time':req_time2
        }
    master_path = "LogMaster/FaceRecognition_Login-Master.xlsx"
    write_excel(master_path, master_data)
    return ret_dict

# @FastAPI_app.post("/api/checkPhotos/",)
# async def get_two_Images(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...),):
#     print("-------- In url get_two_Images func --------")
#     t = time.localtime()
#     img_file_1 = file1.filename
#     img_file_2 = file2.filename
#     # print(img_file_1, img_file_2)
#     timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
#     img_file_1 = f"{timestamp}_{img_file_1}"
#     img_file_2 = f"{timestamp}_{img_file_2}"
#     img_file_1_path = os.path.join("Uploads", img_file_1)
#     img_file_2_path = os.path.join("Uploads", img_file_2)
#     #print(img_file_1_path, img_file_2_path)
#     # if not os.path.exists(Input_folderPath):
#     #     os.makedirs(Input_folderPath)
#     # if not os.path.exists(Output_folderPath):
#     #     os.makedirs(Output_folderPath)
#     with open( img_file_1_path, "wb") as buffer:
#         shutil.copyfileobj(file1.file, buffer)
#     with open( img_file_2_path, "wb") as buffer:
#         shutil.copyfileobj(file2.file, buffer)
#     results, result_d = face_recg(img_file_1_path, img_file_2_path)
#     print(results, result_d)
#     data = {"Image_file1_path": img_file_1_path,
#             "Image_file2_path":img_file_2_path,
#             "Note":"Distance near 0 - similar faces, disance near 1 - different faces",
#             "similar_flag":str(results[0]), 
#             "results_dist":float(result_d[0])
#             }    
#     return data

    # t = time.localtime()
    # img_file_1 = img_file_url   #file1.filename
    # # print(img_file_1, img_file_2)
    # timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
    # img_file_1 = f"{timestamp}_{img_file_1}"
    # img_file_1_path = os.path.join("Uploads", img_file_1)

    # with open( img_file_1_path, "wb") as buffer:
    #     shutil.copyfileobj(file1.file, buffer)