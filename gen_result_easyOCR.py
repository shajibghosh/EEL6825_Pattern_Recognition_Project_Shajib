#  -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import shutil 
import zipfile 
import time
from PIL import Image

input_img_directory = './ICDAR2015/test_img/' 
input_gt_directory = './ICDAR2015/test_gt/'

result_zip_dir = './CLEval/result/' 

root = os.getcwd()

result_dir = os.path.join(root, 'results_easyOCR_ICDAR2015')
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)

result_img_dir = os.path.join(result_dir, 'output_img')
if not os.path.exists(result_img_dir):
	os.makedirs(result_img_dir)

result_csv_dir = os.path.join(result_dir, 'output_csv')
if not os.path.exists(result_csv_dir): 
	os.makedirs(result_csv_dir)

result_txt_dir = os.path.join(result_dir, 'output_txt')
if not os.path.exists(result_txt_dir): 
	os.makedirs(result_txt_dir)
'''
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()
'''
def easy_ocr(image_path):
	base = os.path.basename(image_path)
	image_name = os.path.splitext(base)[0]
	reader = easyocr.Reader(['ch_sim','en'], gpu = False) 
	result = reader.readtext(image_path)
	img = cv2.imread(image_path)
	print("\nEnd-to-End Text Detection and Recognition Results using EasyOCR:\n")
	# loop over the results
	list_of_tuples = [] 
	for (bbox, text, prob) in result:
		# display the OCR'd text and associated probability
		print("[{} , ('{}',{:.7f})]".format(bbox, text, prob))
		# unpack the bounding box
		(tl, tr, br, bl) = bbox
		tl = (int(tl[0]), int(tl[1]))
		tr = (int(tr[0]), int(tr[1]))
		br = (int(br[0]), int(br[1]))
		bl = (int(bl[0]), int(bl[1]))
        
		res_tuple = (tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1], text, prob)
		list_of_tuples.append(res_tuple)
		# cleanup the text and draw the box surrounding the text along
		# with the OCR'd text itself
		#text = cleanup_text(text)
		cv2.rectangle(img, tl, br, (0, 255, 0), 2)
		cv2.putText(img, text, (tl[0]-5, tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	    

	with open(os.path.join(result_csv_dir,'result_easyOCR_{}.csv'.format(image_name)),'w', newline='',encoding="utf-8") as out:
		csv_out=csv.writer(out)
		csv_out.writerow(['BBox','Detected Text','Confidence Score'])
		for row in result:
			csv_out.writerow(row)
    
	df_out = pd.DataFrame(list_of_tuples)

	return img, csv_out, df_out

start = time.time()
for files in os.listdir(input_img_directory):
	fname = files.split('.')[0]
	print("\nProcess starting for " + str(fname) + ".....\n")
	out_image, out_csv, out_df = easy_ocr(input_img_directory + files) 
	cv2.imwrite(((result_img_dir + '/' + 'result_easyOCR_{}').format(files)), out_image)
	out_df.to_csv((result_txt_dir + '/' + 'res_{}.txt').format(fname), header=None, index=None, sep=',', mode='a')
	print("Process completed for "+ str(files) + "\n")
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("\nElapsed time : {:0>2} hrs {:0>2} mins {:05.2f} secs. \n".format(int(hours),int(minutes),seconds))

zipped_result = zipfile.ZipFile(os.path.join(result_zip_dir, 'result_easyOCR_IC15.zip'), "w")
for files in os.listdir(result_txt_dir):
    zipped_result.write(os.path.join(result_txt_dir,files), files)
zipped_result.close()

