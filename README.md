# BreastSegmentation
This Repo contains work done to segment Mass from Mammography Images

Link to MIAS Dataset  (http://peipa.essex.ac.uk/info/mias.html)
- Goal : Is to detect the mass and segment it on Mammography Images of DDSM Data
- This Notebook looks at a smaller dataset MIAS and attempts at classifying betweeN Normal vs Cancerous
- This is to explore the type of data , Features used  and also required preprocessing
- VGG Bottle Neck features are used. A Dense Layer with Heavy Regularization is used to learn 
- Idea is to train a separate R CNN to segment the Mass and the same network to classify as Malignant vs Benign

**Pre Processing **
- All Images are enhanced using CLAHE (Contrast Limited Adaptive Histogram Equalization)
  -- https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
- An Adaptive Median Thresholding is performed to Remove the lines, Letters and other Boxes irrelevant to the Breast Image
- Refer to MIASPreprocessing.py

*Note: Manual Input is taken inorder to improve accuracy of Pre Processing*
 
