import cv2
import numpy as np
import matplotlib.pyplot as plt


v_path = "ball.mov"
video = cv2.VideoCapture(v_path)


def findcentre(img, threshold):
    m, n, channels = img.shape
    red_coordinates = []
    image_out = np.zeros((m, n, channels))

    for i in range(0, m):
        for j in range(0, n):
            if img[i][j][2] > 120 and img[i][j][0] < threshold and img[i][j][1] < threshold:
                image_out[i][j][2] = img[i][j][2]
                cord = [i, j]
                red_coordinates.append(cord)

            else:
                image_out[i][j][2] = 0

    x_mid, y_mid = 0, 0
    x, y = [], []
    for i in range(0, (len(red_coordinates))):
        x.append(red_coordinates[i][0])
        y.append(red_coordinates[i][1])

    x_mid = int(min(x) + (max(x) - min(x)) / 2)
    y_mid = int(min(y) + (max(y) - min(y)) / 2)

    img[x_mid][y_mid][0] = 255
    img[x_mid][y_mid][1] = 255
    img[x_mid][y_mid][2] = 255

    print('X = %f, Y = %f' % (int(x_mid), int(y_mid)))

    return [int(x_mid), int(y_mid)], img

if __name__ == '__main__':

        threshold = 50
        centre_list_x = []
        centre_list_y = []
        flag, image = video.read()
        count = 0

        while flag:
            flag, image = video.read()
            if flag == True:
                try:
                    centre_points, img = findcentre(image, threshold)

                    centre_list_x.append(centre_points[0])
                    centre_list_y.append(centre_points[1])
                    count += 1
                except:
                    pass

        x = centre_list_x
        y = centre_list_y

        plot = plt.scatter(centre_list_y, centre_list_x)

        fig, ax = plt.subplots()
        im = cv2.imread("download.png")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = ax.imshow(im)
        ax.plot(centre_list_y, centre_list_x, ls='dotted', linewidth=2, color='white')
        plt.show()