#! /usr/bin/env python

import cv2
import numpy as np
import rospy
import pyrealsense2 as rs
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
        self.camera_info_subscriber = rospy.Subscriber("/camera/",CameraInfo,callback=self.CameraInfo_callback)
        self.dep_publisher = rospy.Publisher("QR_code",code,queue_size=1)
        self.rgb_image = None
        self.depth_image = None
        self.fx_fy_cx_cy = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    def Depth_callback(self,msg):
        self.depth_image = msg
        #bridge=CvBridge()
        #self.depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")

    def Image_callback(self,msg):
        self.rgb_image = msg
        #bridge=CvBridge()
        #self.cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    def CameraInfo_callback(self,msg):
        self.fx_fy_cx_cy[0] = msg.K[0]
        self.fx_fy_cx_cy[1] = msg.K[4]
        self.fx_fy_cx_cy[2] = msg.K[2]
        self.fx_fy_cx_cy[3] = msg.K[5]
        # rospy.loginfo(self.fx_fy_cx_cy)

    def detect(self):
        rate = rospy.Rate(60)
        # pipeline = rs.pipeline()  # 定义流程pipeline
        # config = rs.config()  # 定义配置config
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
        # align_to = rs.stream.color  # 与color流对齐
        # align = rs.align(align_to)
        # frames = pipeline.wait_for_frames()  # 等待获取图像帧
        # aligned_frames = align.process(frames)  # 获取对齐帧
        # # aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        # color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
        # intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        # # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        while not rospy.is_shutdown():
            if self.rgb_image is not None:
                qrcoder = cv2.QRCodeDetector()
                points = qrcoder.detect(self.rgb_image)
                #print(points)
                #cv2.drawContours(self.rgb_image, [np.int32(points)], 0, (0, 0, 255), 2)
                if points[0]:
                    # print("yes")
                    x = (points[1][0,0,0]+points[1][1,0,0]+points[1][2,0,0]+points[1][3,0,0])/4
                    y = (points[1][0,0,1]+points[1][1,0,1]+points[1][2,0,1]+points[1][3,0,1])/4
                    z = (self.depth_image[points[1][0,0,0]][points[1][0,0,1]]+self.depth_image[points[1][1,0,0]][points[1][1,0,1]]+
                        self.depth_image[points[1][2,0,0]][points[1][2,0,1]]+self.depth_image[points[1][3,0,0]][points[1][3,0,1]])/4
                    # print(x,y,z)
                    fx = self.fx_fy_cx_cy[0]
                    fy = self.fx_fy_cx_cy[1]
                    cx = self.fx_fy_cx_cy[2]
                    cy = self.fx_fy_cx_cy[3]
                    code.z = z * 0.001
                    code.x = (x - cx) / fx * code.z
                    code.y = (y - cy) / fy * code.z
                    # print(code.x,code.y,code.z)
                    self.dep_publisher.publish(code)
                else:
                    # print("no")
                    code.z = -1
                    code.x = 0
                    code.y = 0
                    self.dep_publisher.publish(code)
            rate.sleep()


if  __name__ == "__main__":
    c = code_detecter()
    c.detect()
   
