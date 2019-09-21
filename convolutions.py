import numpy as np
from skimage.exposure import rescale_intensity
import argparse
import cv2

def convoolve(image, K):
    #grab the spatial dimensions of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for the output image, taking care to "pad"
    # the borders of the input image so the spatial size (i.e.,
    # width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype='float')

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, pad+iH):
        for x in np.arange(pad, pad+iW):
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]

            # perform convolution
            k = (roi * K).sum()

            # store convolve value in the output
            output[y - pad, x - pad] = k

    # rescale output image to be in rang 0-255
    output = rescale_intensity(output, in_range=(0, 255))
    # convert from float to unit8
    output = (output * 255).astype('uint8')

    return output

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to the input image')

args = vars(ap.parse_args())
smallBlur = np.ones((7, 7), dtype='float') * (1.0 / (7 * 7))
largBlur = np.ones((21, 21), dtype='float') * (1.0 / (21 * 21))

sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype='int')

laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype='int')

sobelX = np.array(([-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]), dtype='int')
sobelY = sobelX.T

emboss = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype='int')

# construct the kernel bank, a list of kernels we’re going to apply
# using both our custom ‘convole‘ function and OpenCV’s ‘filter2D‘
# function
kernalBank = (
    ("small blur", smallBlur),
    ("large blur", largBlur),
    ('sharpen', sharpen),
    ('laplacian', laplacian),
    ('sobel_x', sobelX),
    ('sobel_y', sobelY),
    ('emboss', emboss)
)

#load image and convert it to gray sclae
image = cv2.imread(args['image'])
image = cv2.resize(image,(300, 300), interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernal_name, K) in kernalBank:
    # apply the kernel to the grayscale image using both our custom
    # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print("[INFO] applying {} kernel".format(kernal_name))
    convoolveOutput = convoolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernal_name), convoolveOutput)
    cv2.imshow('{} - opencv'.format(kernal_name), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
