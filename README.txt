#####################################################################################
#####################################################################################
##                                                                                 ##
##          EEL 6825: Pattern Recognition and Intelligent Systems                  ##
##                                                                                 ##
##     An End-to-End System for Detecting and Recognizing Texts in Natural         ##
##                                                                                 ##
##           Scene Images using Deep Convolutional Neural Networks                 ##
##                                                                                 ##
#####################################################################################
#####################################################################################

1. Included files and folders in 'EEL6825_Pattern_Recognition_Project_Shajib' folder: 

        -- [CLEval]
	-- [ICDAR2015]
	-- [fonts]
        -- [sample_images]
        -- e2e_text_det_rec.py
        -- environment.yml
        -- gen_result_easyOCR.py
        -- gen_result_paddleOCR.py
        -- README.txt

2. Setting up the working directory and environment:

      -- make sure that the system has anaconda distribution properly installed 
      -- open the anaconda prompt
      -- set up the working directory by writing the command:
		
		cd [full path to 'EEL6825_Pattern_Recognition_Project_Shajib' folder]

3. Setting up the virtual conda environment:

      -- now set up the virtual conda environment by writing following command:

		conda env create -f environment.yml 

      -- after successfully creating the virtual environment, write the following command:

                conda activate e2e_ocr_shajib

4. Running the code:

      -- for a single image, run the 'e2e_text_det_rec.py' by writing following command:
           	
		python e2e_text_det_rec.py   

		[in the code, set the image path, e.g., 
			img_path = os.path.join(parent_dir, 'sample_images/sample_1.jpg') ]

      -- if you want to generate results for ICDAR2015 test dataset using EasyOCR, 
		run the 'gen_result_easyOCR.py' by writing following command:

		python gen_result_easyOCR.py 
 
      -- if you want to generate results for ICDAR2015 test dataset using paddleOCR, 
		run the 'gen_result_paddleOCR.py' by writing following command:

		python gen_result_paddleOCR.py
	
5. Evaluating and comparing the performance of the end-to-end frameworks:

	-- set up the working directory by writing the command:

		cd [full path to 'EEL6825_Pattern_Recognition_Project_Shajib/CLEval' folder]

	-- if you want to evaluate the performance of EasyOCR framework, write the following command:

		python script.py -g=gt/gt_IC15.zip -s=result/result_easyOCR_IC15.zip --E2E

	-- if you want to evaluate the performance of paddleOCR framework, write the following command:

		python script.py -g=gt/gt_IC15.zip -s=result/result_paddleOCR_IC15.zip --E2E

DEMO:

Please check these YouTube videos for the demonstration of the project:

1. EEL6825 Final Project Presentation - Part 1: 
https://www.youtube.com/watch?v=Gp-EK7EvzKg

2. EEL6825 Final Project Presentation - Part 2:
https://www.youtube.com/watch?v=1t_HsJmGv_Q

ACKNOWLEDGEMENTS:

I would like to thank the owners of the following repositories that inspired 
me to develop this system:

1. https://github.com/JaidedAI/EasyOCR

2. https://github.com/PaddlePaddle/PaddleOCR

3. https://github.com/clovaai/CLEval

######################################################################################
######################################################################################
##										    ##
##					      THE END                               ##
##										    ##
######################################################################################
###################################################################################### 
