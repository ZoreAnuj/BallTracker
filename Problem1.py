import cv2
import numpy as np
import cmath
import matplotlib.pyplot as plt


def Ball_Centre(image):
    y_values, x_values = np.nonzero(image)
    if x_values.size == 0:
        X = np.nan
    else:
        zero_count = np.count_nonzero(x_values == 0)
        nan_count = np.count_nonzero(np.isnan(x_values))
        inf_count = np.count_nonzero(np.isinf(x_values))
        if zero_count > 0 or nan_count > 0 or inf_count > 0:
            X = np.nan
        else:
            X = np.sum(x_values) / x_values.shape[0]

    if y_values.size == 0:
        Y = np.nan
    else:
        zero_count = np.count_nonzero(y_values == 0)
        nan_count = np.count_nonzero(np.isnan(y_values))
        inf_count = np.count_nonzero(np.isinf(y_values))
        if zero_count > 0 or nan_count > 0 or inf_count > 0:
            Y = np.nan
        else:
            Y = np.sum(y_values) / y_values.shape[0]

    return (int(X), int(Y))


video = cv2.VideoCapture('ball.mov')
image = cv2.imread('download.png')

lower_red = np.array([5, 6, 130], dtype="uint8")
upper_red = np.array([60, 60, 255], dtype="uint8")
filtered = cv2.inRange(image, lower_red, upper_red)
output = cv2.bitwise_and(image, image, mask=filtered)

i = 0
cv_image = []
flag, frame = video.read()
flag = True
while flag:
    flag, frame = video.read()
    if flag == True:
        cv_image.append(frame)

img_cp = cv_image[1]
g_img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(g_img, 0, 50, 0)

print("Frames captured: ", len(cv_image), "\n")


coordinates = []
for images in cv_image[:]:
    filtered = cv2.inRange(images, lower_red, upper_red)
    output = cv2.bitwise_and(images, images, mask=filtered)
    g_img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(g_img, 0, 255, 0)

    try:
        cX, cY = Ball_Centre(threshold)
        cv2.circle(images, (cX, cY), 5, (255, 255, 255), -1)
        Coord_Spec = (cX, cY)
        coordinates.append(Coord_Spec)
    except:
        continue

im = plt.imread('download.png')
implot = plt.imshow(im)

# x and y coordinates

x = [i[0] for i in coordinates]
y = [i[1] for i in coordinates]

coordinate_x = np.array(x)
coordinate_y = np.array(y)



# Visualization

plt.scatter(x, y, s=10, marker='*', color="green")


def fit_curve(coordinate_x, coordinate_y, degree=2):
    n = len(coordinate_x)

    X = np.ones((n, degree + 1))
    for j in range(degree):
        X[:, j] = coordinate_x ** (degree - j)

    # Solve with least-squares
    XT_X = np.dot(X.T, X)
    XT_Y = np.dot(X.T, coordinate_y)
    coefficient = np.linalg.solve(XT_X, XT_Y)
    x_fit = np.linspace(coordinate_x.min(), coordinate_x.max(), 1000)
    y_fit = np.zeros_like(x_fit)
    for j in range(degree + 1):
        y_fit += coefficient[j] * x_fit ** (degree - j)

    return x_fit, y_fit, coefficient


x_fit, y_fit, coefficient = fit_curve(coordinate_x, coordinate_y)

[a, b, c] = coefficient

plt.plot(coordinate_x, coordinate_y, '.', label='Data', color='blue')
plt.plot(x_fit, y_fit, label='Fitted Curve', color='green')
plt.legend()
plt.show()

print('Fitted curves Equation is: ')
print('y = %f + (%f)x + (%f)x^2' % (c, b, a), "\n")

y1 = a * (0) + b * (0) + c
y2 = y1 + 300
c = y1 - y2
dis = (b ** 2) - (4 * a * c)
x1 = (-b - cmath.sqrt(dis)) / (2 * a)
x2 = (-b + cmath.sqrt(dis)) / (2 * a)
print('Roots')
print(x1)
print(x2, "\n")

plt.scatter(x,y)
plt.show()

print("Landing Spot X coordiante =",x2)