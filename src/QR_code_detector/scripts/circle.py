import cv2
import numpy as np

kernel = np.ones((20, 20), np.uint8)


def Find(img, name):
    gray = cv2.cvtColor(cv2.blur(img, (2, 2)), cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  #TOPHAT
    ret, th1 = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    markers = []
    contours, hierachy = cv2.findContours(th1, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    if not hierachy is None:
        hierachy = hierachy[0]
        for i in range(len(hierachy)):
            #print(hierachy)
            contourInfo = hierachy[i]
            if contourInfo[2] == -1:  #无子轮廓
                n = 0
                index = -1
                #追溯父轮廓
                while contourInfo[3] != -1:
                    index = contourInfo[3]
                    contourInfo = hierachy[index]
                    n += 1
                    if n == 4:  #找到正确的外轮廓（5层嵌套）
                        M = cv2.moments(contours[index])  #计算矩
                        cx = int(M['m10'] / M['m00'])  #计算重心
                        cy = int(M['m01'] / M['m00'])  #计算重心
                        markers.append([cx, cy])
                        # cv2.line(img, (int(cx), int(cy)), (int(cx), int(cy)), (
                        #     0,
                        #     255,
                        # ), 3)
                        # cv2.drawContours(img, contours, index, (0, 255, 0), 5)
                        break


    #print(markers)
    #cv2.imshow('img' + name, img)
    #print(markers_filtrated)
    return markers

if __name__ == "__main__":
    while True:
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        Find(frame, '')
        cv2.namedWindow('img')
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
