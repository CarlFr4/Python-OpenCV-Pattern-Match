import myutils as myu
import scipy.ndimage as nd
import numpy as np
import cv2

#path to the 'search image' 1 and 'camera image' 2
path1 = 'C:/Users/carl.friedrich/Pictures/1.jpg'
path2 = 'C:/Users/carl.friedrich/Pictures/2.jpg'
#assign the images
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
imgrec = cv2.imread(path2); imgfinal2 = cv2.imread(path2)
#grayscale the images
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
imgrec = cv2.cvtColor(imgrec, cv2.COLOR_BGR2GRAY)
#increase image contrast
img1 = cv2.convertScaleAbs(img1, 3, 3)
img2 = cv2.convertScaleAbs(img2, 3, 3)
imgrec = cv2.convertScaleAbs(imgrec, 3, 3)
#set a threshold for determination of a match
threshold = .43
#set rotation angle of the search image and rotate the search image by 'rotangle' degrees
rotangle = -2.5
imgrotated = nd.rotate(img1, rotangle, mode='constant', cval = 255)
#find the width and height of the rotated search image
w = imgrotated.shape[1]
h = imgrotated.shape[0]
#slide the rotated search image across the camera image (left -> right, top -> bottom) and calculate the match
myresult = cv2.matchTemplate(img2, imgrotated, cv2.TM_CCOEFF_NORMED)
#find the x and y locations where the match is greater than or equal to the threshold, creating a location array
yloc, xloc = np.where(myresult >= threshold)
#draw a rectangle of the appropriate width and height at each of the x,y locations
for (x, y) in zip(xloc, yloc):
    cv2.rectangle(imgrec, (x, y), (x + w, y + h), (0,255,255), 1)
#finding All Contours
imgrec = cv2.convertScaleAbs(imgrec, 3, 3)
ret, imgrec = cv2.threshold(imgrec, 0, 255, cv2.THRESH_BINARY)
imgrec = cv2.bitwise_not(imgrec)
contours, hierarchy = cv2.findContours(imgrec, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
imgfinal2 = cv2.drawContours(imgfinal2, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS
#resize the images
img1 = cv2.resize(imgrotated, None, fx=1.1, fy=1.1)
img2 = cv2.resize(img2, None, fx=0.2, fy=0.2)
imgfinal2 = cv2.resize(imgfinal2, None, fx=0.2, fy=0.2)
#show the images
cv2.imshow('Modified Search Image', img1)
cv2.imshow('Modified Camera Image', img2)
cv2.imshow('Final Image', imgfinal2)
#wait for a key press and destroy all windows
cv2.waitKey()
cv2.destroyAllWindows()