#! /usr/bin/env python

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from QR_code_detector.msg import code
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int8

class code_detecter:
    def __init__(self) :
        self.node = rospy.init_node("detecter")
        self.rgb_subscriber = rospy.Subscriber("/camera/color/image_raw",Image,callback=self.Image_callback)
        self.dep_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,callback=self.Depth_callback)
        self.camera_info_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo,callback=self.CameraInfo_callback)
        self.dep_publisher = rospy.Publisher("QR_code",code,queue_size=1)
        self.rgb_image = None
        self.depth_image = None
        self.fx_fy_cx_cy = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
        self.bridge=CvBridge()
    def Depth_callback(self,msg):
        # self.depth_image = msg
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def Image_callback(self,msg):
        # self.rgb_image = msg
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def CameraInfo_callback(self,msg):
        self.fx_fy_cx_cy[0] = msg.K[0]
        self.fx_fy_cx_cy[1] = msg.K[4]
        self.fx_fy_cx_cy[2] = msg.K[2]
        self.fx_fy_cx_cy[3] = msg.K[5]
        # rospy.loginfo(self.fx_fy_cx_cy)

    def detect(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            if self.depth_image is not None:
                qrcoder = cv2.QRCodeDetector()
                points = qrcoder.detect(self.rgb_image)
                # print(points)
                # print(self.depth_image[0])
                # print(len(self.depth_image[0]))

                #cv2.drawContours(self.rgb_image, [np.int32(points)], 0, (0, 0, 255), 2)
                if points[0]:
                    # print("yes")
                    x = (points[1][0,0,0]+points[1][0,1,0]+points[1][0,2,0,]+points[1][0,3,0])/4
                    y = (points[1][0,0,1]+points[1][0,1,1]+points[1][0,2,1]+points[1][0,3,1])/4
                    #print(y)
                    # print(self.depth_image[int(points[1][0,0,1])][int(points[1][0,1,0])])
                     #z = (self.depth_image[int(points[1][0,0,1])][int(points[1][0,0,0])]+self.depth_image[int(points[1][0,1,1])][int(points[1][0,1,0])]+
                         #self.depth_image[int(points[1][0,2,1])][int(points[1][0,2,0])]+self.depth_image[int(points[1][0,3,1])][int(points[1][0,3,0])])/4
                    z=self.depth_image[int(y)][int(x)]
                    #print(x,y,z)
                    fx = self.fx_fy_cx_cy[0]
                    fy = self.fx_fy_cx_cy[1]
                    cx = self.fx_fy_cx_cy[2]
                    cy = self.fx_fy_cx_cy[3]
                    c=code()
                    c.flag = True
                    c.z = z * 0.001
                    c.x = (x - cx) / fx * c.z
                    c.y = -((y - cy) / fy * c.z)
                    # print(c.y)
                    # print(c.x,c.y,c.z)
                    self.dep_publisher.publish(c)
                else:
                    # print("no")
                    c=code()
                    c.flag = False
                    c.z = -1
                    c.x = 0
                    c.y = 0
                    self.dep_publisher.publish(c)
                    pass
            rate.sleep()


if  __name__ == "__main__":
    c = code_detecter()
    c.detect()
   
