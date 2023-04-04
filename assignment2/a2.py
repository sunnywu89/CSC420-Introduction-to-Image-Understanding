import numpy as np
import math
from scipy import signal
from skimage import io
import matplotlib.pyplot as plt

def q1a_magnitude_of_gradient(image):
    # using sobel fiter from the slide for egde detection
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = signal.convolve2d(image, Mx, mode='same')
    gradient_y = signal.convolve2d(image, My, mode='same')
    magnitude_of_gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    return magnitude_of_gradient

def q1b_find_path(gradient):
    num_row, num_col = gradient.shape
    # stores parent's index
    path_table = [[0 for _ in range(num_col)] for _ in range(num_row)]
    # stores the min energy up until this index
    energy_table = np.zeros((num_row, num_col))
    energy_table[0] = gradient[0]
    min_energy = math.inf
    min_index = []
    
    # fill up the min enertgy table and path table
    for i in range(1, num_row):
        for j in range(num_col):
            if j == 0:
                min_energy = min(energy_table[i-1][j], energy_table[i-1][j+1])
                min_index = [i-1, j] if energy_table[i-1][j] == min_energy else [i-1, j+1]
            elif j == num_col-1:
                min_energy = min(energy_table[i-1][j], energy_table[i-1][j-1])
                min_index = [i-1, j] if energy_table[i-1][j] == min_energy else [i-1, j-1]
            else:
                min_energy = min(energy_table[i-1][j], energy_table[i-1][j-1], energy_table[i-1][j+1])
                min_index = [i-1, j] if energy_table[i-1][j] == min_energy else [i-1, j-1]
                if min_index != [i-1, j]:
                    min_index = [i-1, j-1] if energy_table[i-1][j-1] == min_energy else [i-1, j+1]
            energy_table[i][j] = min_energy + gradient[i][j]
            path_table[i][j] = min_index

    # get the full path
    last_row = energy_table[num_row-1]
    last_row_index = np.where(last_row == np.amin(last_row))[0][0]
    parent = [num_row-1, last_row_index]
    min_path = [[num_row-1, last_row_index]]
    curr_row, curr_col = parent[0], parent[1]
    while curr_row > 0:
        parent = path_table[curr_row][curr_col]
        min_path.append(parent)
        curr_row, curr_col = parent[0], parent[1]
    return min_path

def q1c_remove_one_path(img, min_energy_path):
    img = img.tolist()
    for i in range(len(min_energy_path)):
        row, col = min_energy_path[i]
        img[row].pop(col)
    img = np.array(img)
    return img

def q1d_remove_paths(img_path, num_remove):
    img = io.imread(img_path, as_gray=True)
    while num_remove:
        gradient = q1a_magnitude_of_gradient(img)
        path = q1b_find_path(gradient)
        curr_img = q1c_remove_one_path(img, path)
        img = curr_img
        num_remove -= 1
    print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.gray()
    plt.show()


if __name__ == "__main__":
    # img = io.imread("./waldo.png", as_gray=True)
    # img = io.imread("./template.png", as_gray=True)
    # print(type(img))
    # g = q1a_magnitude_of_gradient(img)
    # path = q1b_find_path(g)
    # # print(path)
    # q1c_remove_one_path(img, path)
    q1d_remove_paths("./template.png", 15)