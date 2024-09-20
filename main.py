import cv2
import numpy as np
import sys
import getopt

def BinaryErosion(image, kernel):
    image = image / 255
    # Get the dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Apply the erosion operation
    for i in range(image_height):
        for j in range(image_width):
            if np.sum(kernel * padded_image[i:i+kernel_height, j:j+kernel_width]) == np.sum(kernel):
                output_image[i, j] = 255  # Set to 255 for white in the output image

    return output_image

def BinaryDilation(image, kernel):
    # Get the dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Apply the dilation operation
    for i in range(image_height):
        for j in range(image_width):
            if np.sum(kernel * padded_image[i:i+kernel_height, j:j+kernel_width]) > 0:
                output_image[i, j] = 255

    return output_image

def BinaryHitOrMiss(img, kernel):
    fg_kernel = kernel.copy()
    bg_kernel = 1 - kernel

    # Hit-or-miss transform
    img_fg = BinaryErosion(img, bg_kernel)
    img_bg = BinaryErosion(1 - img, bg_kernel)
    #result = cv2.bitwise_and(img_fg, img_bg)
    result = img_fg - img_bg
    return result

def BinaryThinning(img, kernel):
    hit_or_miss = BinaryHitOrMiss(img, kernel)
    thinned = img - hit_or_miss
    return thinned

def BoundaryExtraction(img, kernel):
    eroded = BinaryErosion(img, kernel)
    boundary = img - eroded
    return boundary

def GrayscaleDilation(img, kernel):
    # Get the dimensions of the image and the kernel
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)))

    # Initialize the output image
    output_image = np.zeros_like(img)

    # Apply the dilation operation
    for i in range(img_height):
        for j in range(img_width):
            output_image[i, j] = np.max(padded_image[i:i+kernel_height, j:j+kernel_width])

    return output_image

def GrayscaleErosion(img, kernel):
    # Get the dimensions of the image and the kernel
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)))

    # Initialize the output image
    output_image = np.zeros_like(img)

    # Apply the erosion operation
    for i in range(img_height):
        for j in range(img_width):
            output_image[i, j] = np.min(padded_image[i:i+kernel_height, j:j+kernel_width])

    return output_image

def GrayscaleClosing(img, kernel):
    dilated = GrayscaleDilation(img, kernel)
    closed = GrayscaleErosion(dilated, kernel)
    return closed

def GrayscaleBlackHat(image, kernel):
    closed = GrayscaleClosing(image, kernel)
    black_hat = closed - image
    return black_hat

def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None
    
    if mor_op == 'hit_or_miss':
        img_hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        cv2.imshow('OpenCV Hit-or-miss image', img_hit_or_miss)
        cv2.waitKey(wait_key_time)

        img_hit_or_miss_manual = BinaryHitOrMiss(img, kernel)
        cv2.imshow('Manual Hit-or-miss image', img_hit_or_miss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hit_or_miss_manual
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img, kernel)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = BinaryThinning(img, kernel)
        cv2.imshow('Manual thinning image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
        
    elif mor_op == 'boundary_extraction':
        erode = cv2.erode(img, kernel)
        img_boundary = img - erode
        cv2.imshow('OpenCV boundary extraction image', img_boundary)
        cv2.waitKey(wait_key_time)
        
        img_boundary_manual = BoundaryExtraction(img, kernel)
        cv2.imshow('Manual boundary extraction image', img_boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_manual
        
    elif mor_op == 'black_hat':
        img_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV black-hat image', img_blackhat)
        cv2.waitKey(wait_key_time)
        
        img_black_hat_manual = GrayscaleBlackHat(img_gray, kernel)
        cv2.imshow('Manual black-hat image', img_black_hat_manual)
        cv2.waitKey(wait_key_time)
        img_out = img_black_hat_manual
        
    if img_out is not None:
        cv2.imwrite(out_file, img_out)
        

def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])