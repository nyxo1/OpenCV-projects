import cv2
import numpy as np


def apply_invisible_cloak(frame, background):
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define red color ranges and create masks
    # instructions said green but the starter code comments said red,
    #  i have worked on red colour here
    lower_red1 = np.array([0, 150, 70])
    upper_red1 = np.array([8, 255, 255])

    lower_red2 = np.array([172, 150, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # combine masks and refine if needed
    # How morphologyEx() works:   (from opencv official documentation and personal understanding)

    # cv2.morphologyEx() applies morphological transformations to the mask.

    # MORPH_OPEN removes small noise (erosion then dilation). In other words, 
    # it removes white noise in the foreground which sort of effects the thickness of foreground.

    # MORPH_CLOSE fills small holes inside the detected region (dilation then erosion).
    # Basically since the erosion shrinks the object, thi will increase object area by dialting it.
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # create inverse mask and isolate cloak area
    mask_inv = cv2.bitwise_not(mask)
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # combine background with current frame
    final_output = cv2.add(cloak_area, visible_area)

    return final_output


cap = cv2.VideoCapture(0)

# capture background (press 'b' to save it)
while True:
    ret, background = cap.read()
    cv2.imshow("Press 'b' to capture background", background)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

while True:
    ret, frame = cap.read()
    output = apply_invisible_cloak(frame, background)
    cv2.imshow("Cloak Effect", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()