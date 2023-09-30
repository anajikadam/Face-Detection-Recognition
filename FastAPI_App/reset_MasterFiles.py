
import pandas as pd


cols = ['TimeStamp','member_id','policy_id','TestImage_URL','faces_found',
        'face_matched','matched_score', 'request_time', 'download_time','face_recog_time']
pd.DataFrame(columns=cols).to_excel("LogMaster/FaceRecognition_Login-Master.xlsx", index=False)

cols = ['TimeStamp','member_id','Image_URL','faces_found','request_time','download_time','face_detect_time']
pd.DataFrame(columns=cols).to_excel("LogMaster/FaceRegister-Master.xlsx", index=False)

