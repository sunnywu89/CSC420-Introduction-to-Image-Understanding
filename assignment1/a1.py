from random import gauss
from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from scipy import ndimage, signal
import time

kernel = np.array([[0, 0.125, 0], [0.5, 0.5, 0.125], [0, 0.5, 0]])
img = io.imread("./waldo.png", as_gray=True)


def q1a_conv(img, kernel):
    
    # conv needs to flip the kernel both directions
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    k = int((kernel.shape[0] - 1) / 2)
    org_size = img.shape
    result = np.zeros((org_size[0], org_size[1]))

    # pad the img to make sure the output remains the same size as input
    img = np.pad(img, (k, k), 'constant', constant_values=(0, 0))
    for i in range(1, org_size[0]+1): 
        for j in range(1, org_size[1]+1): 
            result[i-1][j-1] = np.sum(img[i-k:i+k+1,j-k:j+k+1]*kernel)
    plt.imshow(result, cmap='gray')
    plt.gray()
    
    plt.show()


def q1b_is_separable(kernel):
    u, s, vh = np.linalg.svd(kernel)
    zero = np.zeros(s.shape[0]-1)
    # make sure only one sigular value is non-zero
    if s[0] != 0 and np.allclose(s[1:], zero):
        print("This filter is separable")
        return [True, u[:, 0], s, vh[0, :]]
    return [False]

def q1c_conv_separable(img, kernel):
    # for this question I used kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    separable = q1b_is_separable(kernel)
    if not separable[0]:
        print(separable[0])
        return False
    u, s, vh = separable[1], separable[2], separable[3]
    u = math.sqrt(s[0])*u  # vertical filter
    vh = math.sqrt(s[0])*vh #horizontal filter
    # conv needs to flip the kernel both directions
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    k = int((kernel.shape[0] - 1) / 2)
    org_size = img.shape
    result = np.zeros((org_size[0], org_size[1]))
    img = np.pad(img, (k, k), 'constant', constant_values=(0, 0))
    # apply horizontal filter
    for i in range(1, org_size[0]+1): 
        for j in range(1, org_size[1]+1): 
            result[i-1][j-1] = np.sum(img[i,j-k:j+k+1]*vh)
    # apply vertical filter 
    img = np.pad(result, (k, k), 'constant', constant_values=(0, 0))
    result = np.zeros((org_size[0], org_size[1]))
    for i in range(1, org_size[0]+1): 
        for j in range(1, org_size[1]+1): 
            result[i-1][j-1] = np.sum(img[i-k:i+k+1,j]*u)
    plt.imshow(result, cmap='gray')
    plt.gray()
    plt.show()

    
def q1d_cross_corr_separable(img, kernel):
    # separable version of correlation
    # reusing code for convolution for correlation
    # However, convolution filps the kernel in both directions
    # therefore we need to flip it again before passing into the function so it flips it back
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    q1c_conv_separable(img, kernel)
    return

def q1d_cross_corr(img, kernel):
    # not separable version of correlation
    # reusing code for convolution for correlation
    # However, convolution filps the kernel in both directions 
    # therefore we need to flip it again before passing into the function so it flips it back
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    q1a_conv(img, kernel)
    return



def get_gaussian_filter(kernel_size,sigma):
    x_values = np.linspace(-1* (kernel_size//2), kernel_size//2, kernel_size)
    gaussian = np.zeros(x_values.shape[0])
    for i in range(x_values.shape[0]):
        gaussian[i] = 1 / (sigma * math.sqrt(2*math.pi)) * np.exp(-1*pow(x_values[i],2)/(2*pow(sigma,2)))
    #2d gaussian is the outer product of the 1D gaussian
    kernel = np.outer(gaussian.T, gaussian)
    # normalize
    kernel = kernel / np.sum(kernel)
    return kernel


def q2(size, sigma):
    img = io.imread("./waldo.png")
    gaussian_filter = get_gaussian_filter(size,sigma)
    for i in range(3):
        img[:,:,i] = signal.convolve2d(img[:,:,i], gaussian_filter, mode='same')
    plt.imshow(img)
    plt.show()

def q3a_magnitude_of_gradient(image):
    # using sobel fiter from the slide for egde detection
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = signal.convolve2d(image, Mx, mode='same')
    gradient_y = signal.convolve2d(image, My, mode='same')
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    plt.imshow(magnitude, cmap='gray')
    plt.gray()
    plt.show()

def gradient_for_q4(image):
    # using sobel fiter from the slide for egde detection
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = signal.convolve2d(image, Mx, mode='same')
    gradient_y = signal.convolve2d(image, My, mode='same')
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_direction = np.arctan2(gradient_y, gradient_x) # output value is between [-pi, pi]
    # convert radius to degrees and take care of -ve values
    gradient_dir_degree = np.round(gradient_direction*180 / math.pi)
    gradient_dir_degree = np.where(gradient_dir_degree < 0, gradient_dir_degree+180, gradient_dir_degree)
    return magnitude, gradient_dir_degree

def closer_to(direction):
    if direction < 45/2 or 135+45/2 <= direction <= 180:
        return 0
    elif 90-45/2 >= direction >= 45/2:
        return 45
    elif 135-45/2 >= direction >= 90-45/2:
        return 90
    elif 180 - 45/2 >= direction >= 135-45/2:
        return 135
    return 0

def q4_canny(image):
    magnitude, gradient_dir = gradient_for_q4(image)
    padded = np.pad(magnitude, (1, 1), 'constant', constant_values=(0, 0))

    # non maximum suppression
    for i in range(1, gradient_dir.shape[0]+1):
        for j in range(1, gradient_dir.shape[1]+1):
            if closer_to(gradient_dir[i-1][j-1]) == 0:
                if padded[i][j] < padded[i][j+1] or padded[i][j] < padded[i][j-1]:
                    image[i-1][j-1] = 0
            elif closer_to(gradient_dir[i-1][j-1]) == 45:
                if padded[i][j] < padded[i+1][j-1] or padded[i][j] < padded[i-1][j+1]:
                    image[i-1][j-1] = 0
            elif closer_to(gradient_dir[i-1][j-1]) == 90:
                if padded[i][j] < padded[i+1][j] or padded[i][j] < padded[i-1][j]:
                    image[i-1][j-1] = 0
            elif closer_to(gradient_dir[i-1][j-1]) == 135:
                if padded[i][j] < padded[i-1][j-1] or padded[i][j] < padded[i+1][j+1]:
                    image[i-1][j-1] = 0
    plt.imshow(magnitude, cmap='gray')
    plt.gray()
    plt.show()


def q3b_localization(image, target):
    corr = signal.correlate(image, target)
    corr = corr / (np.linalg.norm(image) * np.linalg.norm(target))
    upper_left = np.where(corr == corr.max())
    for location in zip(upper_left[0], upper_left[1]):
        x, y = location
    upper_left_index = [x, y]
    upper_right_index = [x, y-target.shape[1]-1]
    lower_left_index = [x-target.shape[0]-1, y]
    lower_right_index = [x-target.shape[0]+1, y-target.shape[1]-1]
    plt.imshow(image)
    plt.plot([upper_left_index, lower_left_index, lower_right_index, upper_right_index, upper_left_index], 'r')
    plt.show()
        



if __name__ == "__main__":
    kernel = np.array([[0, 0.125, 0], [0.5, 0.5, 0.125], [0, 0.5, 0]])
    a = np.array([[1,2,1],[2,4,2],[1,2,1]])
    b = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    img = io.imread("./waldo.png", as_gray=True)
    img2 = io.imread("./template.png", as_gray=True)
    q3b_localization(img, img2)
