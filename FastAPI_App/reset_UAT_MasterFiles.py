
import pandas as pd

pd.DataFrame(columns=['member_id','image_encoding', 'file_name']).to_excel("LogMaster/UAT_LocalFaceRegistry.xlsx", index=False)
