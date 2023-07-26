import cv2

if __name__=="__main__":
    img=cv2.imread("/home/julien/robot_dog/src/QR_code_detector/test/1.jpeg")
    img2=cv2.imread("/home/julien/robot_dog/src/QR_code_detector/test/3.png")

    qrcoder = cv2.QRCodeDetector()
    points = qrcoder.detect(img)
    if points[0]:
        print(points)
        print("yes")
        x = (points[1][0,0,0]+points[1][1,0,0]+points[1][2,0,0]+points[1][3,0,0])/4
        y=(points[1][0,0,1]+points[1][1,0,1]+points[1][2,0,1]+points[1][3,0,1])/4
        # z=img2[x][y]
        print(img2)


    else:
        print("no")