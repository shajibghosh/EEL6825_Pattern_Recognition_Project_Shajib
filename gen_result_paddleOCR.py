#  -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shutil 
import zipfile 
import time
from PaddleOCR.paddleocr import PaddleOCR,draw_ocr
from PIL import Image


input_img_directory = './ICDAR2015/test_img/' 
input_gt_directory = './ICDAR2015/test_gt/'

result_zip_dir = './CLEval/result/' 

root = os.getcwd()

result_dir = os.path.join(root, 'results_paddleOCR_ICDAR2015')
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
def paddle_ocr(image_path):

	reader = PaddleOCR(lang='en')   
	result = reader.ocr(image_path, cls=False)
	print("\nEnd-to-End Text Detection and Recognition Results using PaddleOCR:\n")
	
	list_of_tuples = []
	boxes = []
	txts = []
	scores = [] 
	for line in result:
		print(line)

		bbox = line[0]
		text = line[1][0]
		prob = line[1][1]

		boxes.append(bbox)
		txts.append(text)
		scores.append(prob)

		(tl, tr, br, bl) = bbox
		tl = (int(tl[0]), int(tl[1]))
		tr = (int(tr[0]), int(tr[1]))
		br = (int(br[0]), int(br[1]))
		bl = (int(bl[0]), int(bl[1]))
        
		res_tuple = (tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1], text, prob)
		list_of_tuples.append(res_tuple)

	image = Image.open(image_path).convert('RGB')
	im_show = draw_ocr(image, boxes, txts, scores, font_path=os.path.join(root,'PaddleOCR/doc/fonts/simfang.ttf'))
	img_out = Image.fromarray(im_show)
	dict = {'BBox': boxes, 'Detected Text': txts, 'Confidence Score': scores}
	csv_out = pd.DataFrame(dict)
	df_out = pd.DataFrame(list_of_tuples)

	return img_out, csv_out, df_out

start = time.time()

for files in os.listdir(input_img_directory):
	fname = files.split('.')[0]
	print("\nProcess starting for " + str(fname) + ".....\n")
	out_image, out_csv, out_df = paddle_ocr(input_img_directory + files)
	out_csv.to_csv(os.path.join(result_csv_dir, 'result_paddleOCR_{}.csv'.format(fname)), index = False)
	out_image.save(os.path.join(result_img_dir, 'result_paddleOCR_{}.jpg'.format(fname))) 
	out_df.to_csv((result_txt_dir + '/' + 'res_{}.txt').format(fname), header=None, index=None, sep=',', mode='a')
	print("Process completed for "+ str(files) + "\n")

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("\nElapsed time : {:0>2} hrs {:0>2} mins {:05.2f} secs. \n".format(int(hours),int(minutes),seconds))

zipped_result = zipfile.ZipFile(os.path.join(result_zip_dir, 'result_paddleOCR_IC15.zip'), "w")
for files in os.listdir(result_txt_dir):
    zipped_result.write(os.path.join(result_txt_dir,files), files)
zipped_result.close()