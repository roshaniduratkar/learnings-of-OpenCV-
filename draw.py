import cv2 as cv
import numpy as np
'''blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow("blank",blank)'''
image=cv.imread(r'C:\Users\HP\Documents\numberplaterecognition\name.png')
cv.imshow('image',image)

blank=np.zeros(image.shape[:2],dtype='uint8')
cv.imshow("blank",blank)

#drawing square
'''blank[:]=25,200,25
blank[100:500,25:400]=100,100,25
cv.imshow("green",blank)'''

#print blank space of different colors drawing square

'''blank[:]=100,200,100
cv.imshow("blue",blank)
blank[:]=200,200,300
cv.imshow("gr",blank)
blank[:]=500,200,500
cv.imshow("gre",blank)'''

#drawing rectangle
'''cv.rectangle(blank,(100,100),(250,250),(100,50,0),thickness=10)
cv.imshow("rectangle",blank)'''

#filling color in shape
'''cv.rectangle(blank,(0,0),(blank.shape[0]//2,blank.shape[1]//2),(0,255,0),thickness=cv.FILLED)
cv.imshow("angle",blank)'''

#draw circle
#cv.circle(blank,(center from width,center from ht.),diametr,(color),thickness=10)
'''cv.circle(blank,(250,blank.shape[1]//2),100,(0,255,0),thickness=10)
cv.imshow("ale",blank)'''

#draw line
#cv.line(blank,(x,y_starting point),(ending point),(color),thickness=5)
'''cv.line(blank,(0,0),(blank.shape[0]//2,blank.shape[1]//2),(255,505,105),thickness=5)
cv.imshow("line",blank)'''

#writing text
#cv.putText(imagename,"text to be print",(x,y distanse in width and ht.),cv.font,3.0(Z00m level),(color cordinate),thickness of text=10)
cv.putText(blank,"hello roshani",(100,250),cv.QT_FONT_BLACK,3.0,(100,250,50),thickness=10)
cv.imshow("text",blank)

#BASIC FUNCTIONS
 #Color function
'''gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('gr',gray)'''
#blur
'''blur=cv.GaussianBlur(image,(7,7),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)'''
#edge cascade
'''cascade=cv.Canny(image,175,175)
cv.imshow("cas",cascade)'''
#diluting image - giving cascade image as input
'''dilated=cv.dilate(cascade,(7,7),iterations=5)
cv.imshow("dilt",dilated)'''
#eroding--giving dilated image as input
'''eroded=cv.erode(dilated,(7,7),iterations=5)
cv.imshow("Eroded",eroded)'''
#resize
'''resize=cv.resize(image,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow("resized",resize)'''

#cropping
'''crop=image[150:300,500:800]
cv.imshow("cropped",crop)
'''

#translation
'''def translate(image,x,y):
    transmat=np.float32([[1,0,x],[0,1,y]])
    dimension=(image.shape[1],image.shape[0])
    return  cv.warpAffine(image,transmat,dimension)

#-x-->left       #-y-->up       #x-->right      y-->down
translated=translate(image,-100,-100)
cv.imshow("trans",translated)'''


#rotation
'''def rotating(image,angle,point=None):
    (height,width) = image.shape[:2]
    if point is None:
        point = (width//2,height//2)
    rotmat = cv.getRotationMatrix2D(point,angle,1.0)
    dimension = (width,height)
    return cv.warpAffine(image,rotmat,dimension)

rotated=rotating(image,45)
#rotated=rotating(image,-45)#negative for anticlockwise
cv.imshow("rotate",rotated)'''

#fliping
#flip=cv.flip(image,0-flip by ht 1-by width -1 for both)
'''flip=cv.flip(image,0)
cv.imshow("flip",flip)'''

#cropping
#croping=image[ht,width]
'''croping=image[100:300,100:500]
cv.imshow("crop",croping)'''

cv.waitKey(0)
cv.destroyAllWindows()
