from imutils import paths
import argparse
import imutils
import cv2
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='path to image directory')
ap.add_argument('-o', '--output', required=True,
                help='path fr output directory')
args = vars(ap.parse_args())

# get the lis of all images
imagePaths = list(paths.list_images(args['input']))


for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{} ".format(i+1, len(imagePaths)))

    # for each image in image paths try
    # loading the image and convert it to grayscale, then pad the
    # image to ensure digits caught on the border of the image
    # are retained
    try:
        image = cv2.imread(imagePath)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey = cv2.copyMakeBorder(grey, 8, 8, 8, 8, borderType=cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # get all the contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = np.squeeze(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = grey[y - 5:y+h+5, x-5:x + w + 5]
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            if key == ord("'"):
                print('[INFO] ignoring character')
                continue
            # grab the passed key and construct a path
            key = chr(key).upper()
            dirPath = os.path.sep.join([args['output'], key])
            if not os.path.exists(dirPath):
                os.mkdir(dirPath)
            p = os.path.sep.join([dirPath, '{}.png'.format(str(len(os.listdir(dirPath))).zfill(6))])
            cv2.imwrite(p, roi)

    except KeyboardInterrupt:
        print('[INFO] manually leaving the script')
        break
    # an unknown error has occurred for this particular image
    except:
        print('[INFO] skipping image...')

