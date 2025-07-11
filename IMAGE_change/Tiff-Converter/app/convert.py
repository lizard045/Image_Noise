import os
import cv2
from zipfile import ZipFile
from app import app
import time

def convert_jpg(files):
  # create a ZipFile object
  zipObj = ZipFile('sample.zip', 'w')
  # Add multiple files to the zip
  for index, file in enumerate(files):
    path = os.path.join(app.config['UPLOAD_PATH'], file.filename)
    # 產生唯一檔名：原檔名_編號_時間戳
    base_name = file.filename.split('.')[0]
    unique_name = f"{base_name}_{index+1:03d}_{int(time.time())}"
    outfile = os.path.join(app.config['UPLOAD_PATH'], unique_name + '.jpg')
    read = cv2.imread(path)
    # Convert to .jpg
    cv2.imwrite(outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
    zipObj.write(outfile, arcname=unique_name + '.jpg')
  # close the Zip File
  zipObj.close()
  delete()

def convert_png(files):
  # create a ZipFile object
  zipObj = ZipFile('sample.zip', 'w')
  # Add multiple files to the zip
  for index, file in enumerate(files):
    path = os.path.join(app.config['UPLOAD_PATH'], file.filename)
    # 產生唯一檔名：原檔名_編號_時間戳
    base_name = file.filename.split('.')[0]
    unique_name = f"{base_name}_{index+1:03d}_{int(time.time())}"
    outfile = os.path.join(app.config['UPLOAD_PATH'], unique_name + '.png')
    read = cv2.imread(path)
    # Convert to .png
    cv2.imwrite(outfile, read)
    zipObj.write(outfile, arcname=unique_name + '.png')
  # close the Zip File
  zipObj.close()
  delete()

def delete():
  #delete uploads dir files
  dir = app.config['UPLOAD_PATH']
  for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))