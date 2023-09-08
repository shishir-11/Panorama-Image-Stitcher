import cv2
import numpy as np
import glob
import imutils
from PIL import Image

image_paths = glob.glob('images/*.png')
images = []

for image in image_paths:
    img = cv2.imread(image)
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

ref = images[0]
ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
l1, a1, b1 = cv2.split(ref_lab)

height, width = ref.shape[:2]

for i in range(1,len(images)):
    img_match = cv2.createCLAHE(clipLimit=1,tileGridSize=(8,8))
    img_lab = cv2.cvtColor(images[i], cv2.COLOR_BGR2LAB)
    l2, a2, b2 = cv2.split(img_lab)
    # l2 = cv2.GaussianBlur(l2, (3, 3), 1)
    l2 = img_match.apply(l2)

    img_match_lab = cv2.merge((l2, a2, b2))

    images[i] = cv2.cvtColor(img_match_lab, cv2.COLOR_LAB2BGR)

    height_img, width_img = images[i].shape[:2]
    if height_img>height:
        height = height_img


imageStitcher = cv2.Stitcher.create(mode=0)

error,stitched_img = imageStitcher.stitch(images)

if not error:
    cv2.imwrite('stitchedOutput.png', stitched_img)
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window",400,height)
    cv2.imshow('Window', stitched_img)
    cv2.waitKey(0)

    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Window", thresh_img)

    cv2.waitKey(0)

    #
    # kernel = np.ones((5, 5), np.uint8)
    # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    #
    # contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # # crop = stitched_img
    # for cntr in contours:
    #     x, y, w, h = cv2.boundingRect(cntr)
    #     crop = crop[y:y + h, x:x + w]
    #     # show cropped image
    #     cv2.imshow("Window", crop)
    #     cv2.waitKey(0)

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    area = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype='uint8')
    x, y, w, h = cv2.boundingRect(area)
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)

    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    area = max(contours, key=cv2.contourArea)

    cv2.imshow("Window", minRectangle)
    cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(area)
    stitched_img = stitched_img[y:y+h, x:x+w]

    cv2.imshow("Window", stitched_img)
    cv2.waitKey(0)

    fin_image = cv2.GaussianBlur(stitched_img, (3, 3),1)

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    fin_image = cv2.filter2D(fin_image, -1, kernel)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # laplacian = cv2.Laplacian(gray, cv2.F)
    # sharp_image = cv2.convertScaleAbs(laplacian)

     # = cv2.GaussianBlur(fin_image, (0, 0), 3)
    #
    # fin_image = cv2.addWeighted(fin_image, 1.5, blurred, -0.5, 0)

    cv2.imshow("Window", fin_image)

    cv2.waitKey(0)
