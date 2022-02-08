import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import time
#example with denim shorts
#ask the user for the paths of dataset and destination folders
print('remember to use the mac/linux path notation with "/"')

path_folder=input("please insert here the path of the dataset folder yu_vton")

path_result=input("please insert here the path of the destination folder for results")


# creation of array of paths of keypoints from a specif category folder
vec = []
for root, dirs, files in os.walk(os.path.abspath(
        path_folder+"/lower_body/keypoints/denim_denim-bermudas")):
    for file in files:
        # print(os.path.join(root, file))
        vec.append(os.path.join(root, file))


# coordinates

f = open(vec[0])
dic = json.load(f)
print("left hip coordinates")
print(dic['keypoints'][8][:2]) #left hip
print("right hip coordinates")
print(dic['keypoints'][11][:2]) #right hip
print("right knee coordinates")
print(dic['keypoints'][12][:2]) #right knee
print("left knee coordinates")
print(dic['keypoints'][9][:2]) #left knee

#re-ordering the coordinates in the correct way to prepare them for the transformation
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

#PERSPECTIVE TRANSFORM function
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        dic['keypoints'][8][:2],
        dic['keypoints'][11][:2],
        dic['keypoints'][12][:2],
        dic['keypoints'][9][:2]],
        dtype="float32")
    dst=dst*np.array([[0.85,1],[1.15,1],[1.15,1],[0.85,1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, 2 * dst)
    warped = cv2.warpPerspective(image, M, (768, 1024))
    # return the warped image
    return warped

#PATHS
##catalogue image
cat_path = path_folder+"/lower_body/images/denim_denim-bermudas/42742272md_1_f.jpg"
##result 1
dest_path_1=path_result+"/ris_1.jpg"
##mannequin
body_path=path_folder+'/lower_body/images/denim_denim-bermudas/42742272md_0_e.jpg'
##superimposition
dest_path_2=path_result+'combined.png'
##shape
dest_path_3=path_result+'/trasp .png'
#########################
# catalogue image upload
image = cv2.imread(cat_path)
plt.imshow(image)
plt.show()
cv2.destroyAllWindows()

# destination points for the Transformation
##bounding box extraction from catalogue image

image_cat = plt.imread(cat_path).astype(np.uint8)
mask = cv2.cvtColor(image_cat, cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(mask, 250, 255, 0)

plt.imshow(thresh, cmap='gray')
plt.show()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
if len(contours) == 0:
    sys.exit('No contours found')

index = -1
areas = []

for j in range(len(contours)):
    rect = cv2.boundingRect(contours[j])
    area = rect[2] * rect[3]
    areas.append([area])

areas = np.array(areas)
areas = np.argsort(areas, axis=0)

rect = cv2.boundingRect(contours[areas[-2].item()])
x, y, w, h = rect
rect2 = cv2.rectangle(image_cat.copy(), (x, y), (x + w, y + h), (200, 0, 0), 2)
plt.imshow(rect2)
plt.show()

pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32") #coordinate bounding box

# apply the function to the first image in the folder
result = four_point_transform(image, pts)
cv2.imwrite(dest_path_1, result)
plt.imshow(result)
plt.show()
# absoluteFilePaths('/Users/federicopallotti/Desktop/COMPUTER_VISION/CV_PROJECT/yu-vton/lower_body/keypoints/denim_denim-bermudas')

# segment the result
##re-apply the bounding box to the segmented result
image_path = dest_path_1

image_cat = plt.imread(image_path).astype(np.uint8)
mask = cv2.cvtColor(image_cat, cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(mask, 250, 255, 0)

plt.imshow(thresh, cmap='gray')
plt.show()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
if len(contours) == 0:
    # print('No contours found')
    sys.exit('No contours found')

index = -1
areas = []

for j in range(len(contours)):
    rect = cv2.boundingRect(contours[j])
    area = rect[2] * rect[3]
    areas.append([area])

areas = np.array(areas)
areas = np.argsort(areas, axis=0)

rect = cv2.boundingRect(contours[areas[-2].item()])
x, y, w, h = rect
rect2 = cv2.rectangle(image_cat.copy(), (x, y), (x + w, y + h), (200, 0, 0), 2)
plt.imshow(rect2)
plt.show()


###Segmentation result
img = result
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = rect
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_seg = img*mask2[:,:,np.newaxis]
plt.imshow(img_seg),plt.colorbar(),plt.show()

#superimposition with the original image with LINEAR BLEND
background=cv2.imread(body_path)
overlay = img
alpha=0.3
beta=1-alpha
added_image = cv2.addWeighted(background,alpha,overlay,beta,0)
plt.imshow(added_image),plt.show()

cv2.imwrite(dest_path_2, added_image)

#########
#edge detection of the mannequin
path=body_path

##image binarization
image = plt.imread(path).astype(np.uint8)
mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

ret, binarized = cv2.threshold(mask, 250, 255, 0)

plt.imshow(binarized, cmap='gray')
plt.show()

#CLOSING of the shape
##DILATION
img = binarized
negative= cv2.bitwise_not(img)
kernel = np.ones((10, 10), 'uint8')

dilate_img = cv2.dilate(negative, kernel, iterations=1)
plt.imshow( dilate_img)
plt.show()


##EROSION
kernel2=np.ones((50, 50), 'uint8')
closing = cv2.morphologyEx(dilate_img, cv2.MORPH_CLOSE, kernel2)
plt.imshow(closing)
plt.show()


#convert img to grey
#set a thresh
thresh = 100
#get thresholded image
ret,thresh_img = cv2.threshold(binarized, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#put the countour on a transparent background
mask = np.zeros((1024, 768, 4), dtype=np.uint8)
cv2.drawContours(mask, contours, -1, (255, 255, 255, 255), 1)
cv2.imwrite(dest_path_3, mask)
plt.imshow(mask)
plt.show()
#(mask.shape)

#apply the contour on the webcam
#create an overlay image
#resize of the contour to fit into the webcam resolution
scale_percent = 70  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
negative= cv2.bitwise_not(mask)
foreground = cv2.resize(negative, dim, interpolation=cv2.INTER_AREA)
#('Resized Dimensions : ',foreground.shape)

# Open the camera
cap = cv2.VideoCapture(0)
# Set initial value of weights
alpha = 0.5
TIMER = int(15)
k = cv2.waitKey(125)

prev = time.time()

while TIMER >= 0:
    # read the background
    ret, background = cap.read()
    background = cv2.flip(background, 1)

    # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
    added_image = cv2.addWeighted(background[2:718, 372:909, :], alpha, foreground[:, :, :3], 1 - alpha, 0)
    # Change the region with the result
    background[2:718, 372:909] = added_image
    # For displaying current value of alpha(weights)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(background, 'Please place yourself into the template'.format(alpha), (5, 30), font, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Display countdown on each frame
    # specify the font and draw the
    # countdown using puttext
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(background, str(TIMER),
                (200, 250), font,
                7, (0, 255, 255),
                4, cv2.LINE_AA)

    cv2.imshow('a', background)


    cv2.waitKey(125)

    # current time
    cur = time.time()

    # Update and keep track of Countdown
    # if time elapsed is one second
    # than decrease the counter
    if cur - prev >= 1:
        prev = cur
        TIMER = TIMER - 1

else:
    ret, img_pic = cap.read()
    cv2.imwrite(path_result+'/camera.jpg', img_pic)

    # Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()


#Sumperimposition with original image


overlay = img_seg
scale_percent = 70  # percent of original size
width = int(overlay.shape[1] * scale_percent / 100)
height = int(overlay.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
foreground = cv2.resize(overlay, dim, interpolation=cv2.INTER_AREA)
background = cv2.flip(background, 1)

alpha=0.35
beta=1-alpha
final_result = cv2.addWeighted(background[2:718, 372:909, :], alpha, foreground[:, :, :3], 1 - alpha, 0)
#plt.imshow(final_result),plt.show()
cv2.imwrite(path_result+'/final_result.jpg', final_result)
#print the output
cv2.imshow('you look good!', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


