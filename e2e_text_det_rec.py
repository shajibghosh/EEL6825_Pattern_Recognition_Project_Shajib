import os
import sys
import easyocr
from PIL import Image
import numpy as np
import cv2
import csv 
import pandas as pd 
import warnings
import time 
from PaddleOCR.paddleocr import PaddleOCR,draw_ocr
warnings.filterwarnings("ignore")


print("\nPlease choose any of the following 2 methods: \n")
print("\n1. EasyOCR [Detection: CRAFT, Recognition: CRNN].\n")
print("\n2. PaddleOCR [Detection:  Differential Binarization (DB), Recognition: CRNN].\n")

choice = int(input("\nEnter your choice (1 or 2): \n")) 
print("\nGenerating Results......\n") 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
parent_dir = os.getcwd()
img_output_dir = os.path.join(parent_dir, 'image_results')
if not os.path.exists(img_output_dir):
	os.makedirs(img_output_dir)
csv_output_dir = os.path.join(parent_dir, 'csv_results')
if not os.path.exists(csv_output_dir):
	os.makedirs(csv_output_dir)
img_path = os.path.join(parent_dir, 'sample_images/sample-1.jpg')    # Define the path of the sample image file 
base_name =  os.path.basename(img_path)
if base_name.endswith('.jpg'):
	sample_name = base_name.split('.jpg')[0]
elif base_name.endswith('.png'):
	sample_name = base_name.split('.png')[0]

def easy_ocr(image_path, name_sample):
    
	reader = easyocr.Reader(['ch_sim','en'], gpu = False) 
	result = reader.readtext(image_path)
	img = cv2.imread(image_path)
	print("\nEnd-to-End Text Detection and Recognition Results using EasyOCR:\n")
	# loop over the results
	for (bbox, text, prob) in result:
		# display the OCR'd text and associated probability
		print("[{} , ('{}',{:.7f})]".format(bbox, text, prob))
		# unpack the bounding box
		(tl, tr, br, bl) = bbox
		tl = (int(tl[0]), int(tl[1]))
		tr = (int(tr[0]), int(tr[1]))
		br = (int(br[0]), int(br[1]))
		bl = (int(bl[0]), int(bl[1]))
		# cleanup the text and draw the box surrounding the text along
		# with the OCR'd text itself
		#text = cleanup_text(text)
		cv2.rectangle(img, tl, br, (0, 255, 0), 2)
		cv2.putText(img, text, (tl[0]-5, tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

	with open(os.path.join(csv_output_dir,'result_easyOCR_{}.csv'.format(name_sample)),'w', newline='',encoding="utf-8") as out:
		csv_out=csv.writer(out)
		csv_out.writerow(['BBox','Detected Text','Confidence Score'])
		for row in result:
			csv_out.writerow(row)
	return img, csv_out

def paddle_ocr(img_path):
	ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
	result = ocr.ocr(img_path, cls=False)
	print("\nEnd-to-End Text Detection and Recognition Results using PaddleOCR:\n")
	for line in result:
		print(line)

	image = Image.open(img_path).convert('RGB')
	boxes = [line[0] for line in result]
	txts = [line[1][0] for line in result]
	scores = [line[1][1] for line in result]
	im_show = draw_ocr(image, boxes, txts, scores, font_path=os.path.join(parent_dir,'PaddleOCR/doc/fonts/simfang.ttf'))
	img_out = Image.fromarray(im_show)
	dict = {'BBox': boxes, 'Detected Text': txts, 'Confidence Score': scores}
	csv_out = pd.DataFrame(dict) 
	return img_out, csv_out 


if choice == 1:
	start1 = time.time()
	out_image, out_csv = easy_ocr(img_path, sample_name)
	cv2.imwrite(os.path.join(img_output_dir,'result_easyOCR_{}.jpg'.format(sample_name)),out_image)
	print("\nResult image file is saved successfully.")
	print("\nFull path to the result image directory: {}".format(img_output_dir))
	print("\nResult csv file is saved successfully.")
	print("\nFull path to the result csv directory: {}".format(csv_output_dir))
	end1 = time.time()
	hours1, rem1 = divmod(end1-start1, 3600)
	minutes1, seconds1 = divmod(rem1, 60)
	print("\nElapsed time : {:0>2} hrs {:0>2} mins {:05.2f} secs. \n".format(int(hours1),int(minutes1),seconds1))
    

elif choice == 2:
	start2 = time.time()
	out_image, out_csv = paddle_ocr(img_path)
	out_csv.to_csv(os.path.join(csv_output_dir, 'result_paddleOCR_{}.csv'.format(sample_name)), index = False)
	out_image.save(os.path.join(img_output_dir, 'result_paddleOCR_{}.jpg'.format(sample_name))) 
	print("\nResult image file is saved successfully.")
	print("\nFull path to the result image directory: {}".format(img_output_dir))
	print("\nResult csv file is saved successfully.")
	print("\nFull path to the result csv directory: {}".format(csv_output_dir))
	end2 = time.time()
	hours2, rem2 = divmod(end2-start2, 3600)
	minutes2, seconds2 = divmod(rem2, 60)
	print("\nElapsed time : {:0>2} hrs {:0>2} mins {:05.2f} secs. \n".format(int(hours2),int(minutes2),seconds2))

else:
    print("\nInvalid choice. Please try again.\n") 

print("\nProcess completed.\n")