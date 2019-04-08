import cv2
import numpy as np
import time
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
from math import sqrt

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("asd",gray_hist_mask_image)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, cv2.THRESH_BINARY)

    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cnt = max(cont, key = lambda x: cv2.contourArea(x))
    #cv2.drawContours(hist_mask_image, cont, -1, (0,255,0), 3)
    #cv2.imshow("cont1",hist_mask_image);
    
    
    #cv2.drawContours(hist_mask_image, [cnt], -1, (0,255,0), 3)
    #cv2.imshow("cont2",hist_mask_image);

    return cont


def max_contour(contour_list):
    return max(contour_list, key = lambda x: cv2.contourArea(x))


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",hsv_frame)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]
    #bgr = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR)
    #cv2.imshow("a",bgr)
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)


    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    #cv2.imshow("stel",thresh)

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        
        #start
        s = defects[:, 0][:, 0]

        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(1,len(traverse_point)):
            start = traverse_point[i-1]
            end = traverse_point[i]
            cv2.line(frame,start,end,[0,0,255],5)
            #cv2.circle(frame, traverse_point[i], 3, [0, 0, 255], -1)


def manage_image_opr(frame, hand_hist,prev_point):
    hist_mask_image = hist_masking(frame, hand_hist)
    #cv2.imshow("asd",hist_mask_image)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    epsilon = 0.0005*cv2.arcLength(max_cont,True)
    approx= cv2.approxPolyDP(max_cont,epsilon,True)

    #cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
    #cv2.imshow("cont",frame);

    cnt_centroid = centroid(max_cont)
    #cv2.circle(frame, cnt_centroid, 10, [123, 56, 12], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        
        defects = cv2.convexityDefects(max_cont, hull)
        """
        print(defects.shape)
        for i in range(defects.shape[0]):
          s,e,f,d = defects[i,0]
          start = tuple(max_cont[s][0])
          end = tuple(max_cont[e][0])
          far = tuple(max_cont[f][0])
          cv2.line(frame,start,end,[0,100,100],2)
          cv2.circle(frame,far,2,[0,0,255],-1)

        cv2.imshow('img',frame)
        """
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        distance = 0
        if prev_point != None:
          x1,y1 = prev_point
          x2,y2 = far_point
          distance = sqrt((x1-x2)**2 + (y1-y2)**2)
        
        if distance > 150:
          far_point = prev_point


        #print(far_point)

        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        #cv2.circle(frame, far_point, 10, [0, 0, 255], -1)
        #if len(traverse_point) < 20:
        #print(distance)
        

        #else:
            #traverse_point.pop(0)
        traverse_point.append(far_point)

        draw_circles(frame, traverse_point)

        return far_point
    return None


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(-1)#WebcamVideoStream(-1).start()
    frames = 0
    start = time.time()
    prev_point = None
    while True:

        

        pressed_key = cv2.waitKey(1)
        _,frame = capture.read()
        #frame = imutils.resize(frame, width=500)

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            prev_point = manage_image_opr(frame, hand_hist,prev_point)



        else:
            #pass
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key & 0xFF == ord('q'):
            break
        
        
        


        

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()