import os
import numpy as np
import sys
import matplotlib.image as mpimg

# change this property
#NOMEROFF_NET_DIR = os.path.abspath('../../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
#MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
#MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

#sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
#from NomeroffNet import  filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync
from OptionsDetector import OptionsDetector

# Initialize npdetector with default configuration file.
#nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
#nnet.loadModel("latest")

#rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
#textDetector = TextDetector.get_static_module("kz")()
#textDetector.load("latest")

# Detect numberplate
img_path = 'images/long_kz.jpg'
img = mpimg.imread(img_path)
img = [img]

#NP = nnet.detect([img])

# Generate image mask.
#cv_img_masks = filters.cv_img_mask(NP)

# Detect points.
#arrPoints = rectDetector.detect(cv_img_masks)
#zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
#print("zones")
#print(zones)

# find standart
regionIds, stateIds, countLines = optionsDetector.predict(img)
regionNames = optionsDetector.getRegionLabels(regionIds)
 
# find text with postprocessing by standart  
#textArr = textDetector.predict(zones)
#textArr = textPostprocessing(textArr, regionNames)
print("Region names: "+str(regionNames))
# ['JJF509', 'RP70012']