from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import cv2

# designate image path here
img_path = 'E:\\Users\\Chase\\Documents\\source\\python_projects\\registration\\forest\\'
IX_path = img_path + 'forest_20160217_L1A0001416073-GF2_PMS1_E116_r8c8_liner002.tif'
IY_path = img_path + 'forest_20161228_L1A0002076029-GF2_PMS2_E115_r8c8_liner002.tif'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

#initialize
reg = Registration.CNN()
#register
X, Y, Z = reg.register(IX, IY)
#generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)
cb = checkboard(IX, registered, 11)

plt.subplot(131)
plt.title('reference')
plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title('registered')
plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.title('checkboard')
plt.imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))

plt.show()






    
