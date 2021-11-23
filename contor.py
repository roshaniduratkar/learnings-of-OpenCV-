import cv2 as cv
import numpy as np
import matplotlib.pyplot as pb

image=cv.imread(r'C:\Users\HP\Documents\numberplaterecognition\images2.jpg')
cv.imshow('image',image)


'''pb.imshow(image)#showing image in plot it shows image in rgb format
pb.show()'''

gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('gr',gray)

'''blur=cv.GaussianBlur(image,(7,7),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)'''
#edge cascade- show edges
'''cascade=cv.Canny(image,175,175)
cv.imshow("cas",cascade)'''

#contour-
#contour-list of all corner of contour,heirarchy=cv.findContours(cascade,
# cv.RETR_LIST-all contour,cv.CHAIN_APPROX_NONE-)
'''contour,heirarchy=cv.findContours(cascade,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(len(contour))'''
#we can reduce contour by giving blur image

#threshholding-binarise image and printing contour
'''ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow("tresh",thresh)
contour,heirarchy=cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(len(contour))
'''

#drawing contours on blank image
blank=np.zeros(image.shape,dtype='uint8')
cv.imshow("blank",blank)

ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow("tresh",thresh)

contour,heirarchy=cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(len(contour))

dr=cv.drawContours(blank,contour,-1,(0,100,225),2)
cv.imshow("drawing contour",dr)

#color spacing
'''hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)   #bgr to hsv
cv.imshow('hsv',hsv)
lab=cv.cvtColor(image,cv.COLOR_BGR2LAB)   #bgr to lab
cv.imshow('lab',lab)
rgb=cv.cvtColor(image,cv.COLOR_BGR2RGB)   #bgr to RGB
cv.imshow('RGB',rgb)
pb.imshow(rgb)#showing image in plot
pb.show()'''


#color channels
'''b,g,r=cv.split(image)
cv.imshow("green",g)
cv.imshow("blue",b)
cv.imshow("red",r)

print(image.shape)
print(b.shape)
print(g.shape)
print(r.shape)
merg=cv.merge([b,g,r])
cv.imshow("merg",merg)

blank=np.zeros(image.shape[:2],dtype='uint8')
cv.imshow("blank",blank)
#merging b,g,r to blank image
blue=cv.merge([b,blank,blank])
red=cv.merge([blank,blank,r])
green=cv.merge([blank,g,blank])
cv.imshow("gre",green)
cv.imshow("blu",blue)
cv.imshow("rd",red)
'''

#blurring methods

#averging---finding pixel's intensity
'''avreging=cv.blur(image,(3,3))
cv.imshow("aver",avreging)
#guassian blur
gblur=cv.GaussianBlur(image,(3,3),0)#more natural than averging
cv.imshow("gblur",gblur)
#median blur
mblur=cv.medianBlur(image,3)#more natural than averging
cv.imshow("mblur",mblur)

#bilaterl
bilateral=cv.bilateralFilter(image,50,35,50)
cv.imshow("bilatr",bilateral)'''

#bitwise operations
'''blank=np.zeros((400,400),dtype='uint8')
cv.imshow("blank",blank)
rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),225,-1)
circle=cv.circle(blank.copy(),(200,200),200,200,-1)
cv.imshow("rect",rectangle)
cv.imshow("cirle",circle)

#bitwise AND--intersecting region
bit_and=cv.bitwise_and(rectangle,circle)
cv.imshow("bit_and",bit_and)
#bitwise or
bit_or=cv.bitwise_or(rectangle,circle)
cv.imshow("bit_or",bit_or)
#bitwise xor
bit_xor=cv.bitwise_xor(rectangle,circle)
cv.imshow("bit_xor",bit_xor)
#bitwise not
bit_not=cv.bitwise_not(circle)
cv.imshow("bit_not",bit_not)
'''


#masking
'''blank=np.zeros(image.shape[:2],dtype='uint8')
cv.imshow("blank",blank)

mask_cir=cv.circle(blank,(image.shape[1]//2,image.shape[0]//2),100,255,-1)
cv.imshow("mask",mask_cir)

mask_rect=cv.rectangle(blank,(100,100),(250,250),(100,50,0),-1)
cv.imshow("mask_rect",mask_rect)

masked_cir=cv.bitwise_and(image,image,mask=mask_cir)
cv.imshow("masked_cir",masked_cir)

masked_rect=cv.bitwise_and(image,image,mask=mask_rect)
cv.imshow("masked_rect",masked_rect)'''

#Histogram
'''gray_gist=cv.calcHist([image],[1],None,[256],[0,256])
pb.figure()
pb.title("histo")
pb.xlabel("bin")
pb.ylabel("no. of pixel")
pb.plot(gray_gist)
pb.xlim([0,256])
pb.show()'''

#color histogram
'''pb.figure()
pb.title("histo")
pb.xlabel("bin")
pb.ylabel("no. of pixel")
color=('b','g','r')
for i,col in enumerate(color):
    hist=cv.calcHist([image],[i],None,[256],[0,256])
    pb.plot(hist,color=col)
    pb.xlim([0, 256])
pb.show()'''

#thersholding
 #simpe thersholding
'''thresholding,thresh=cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow("simple tresh",thresh)

thresholding,thre_inv=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow("tresh",thre_inv)'''
 #adaptive thersholding
'''adaptive_tre=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)
cv.imshow("adap",adaptive_tre)

ad_tre=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow("adp",ad_tre)
'''

# Edge detecting

#laplacian
'''lap=cv.Laplacian(gray,cv.CV_32F)
lap=np.uint8(np.absolute(lap))
cv.imshow("laplacian",lap)'''

#sobel
'''sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobaly=cv.Sobel(gray,cv.CV_64F,0,1)
cv.imshow("sobelx",sobelx)
cv.imshow("sibely",sobaly)
combind=cv.bitwise_or(sobelx,sobaly)
cv.imshow("comb",combind)

canny=cv.Canny(gray,50,300)
cv.imshow("canny",canny)'''




cv.waitKey(0)
cv.destroyAllWindows()