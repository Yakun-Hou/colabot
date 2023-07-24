import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int8

class code_detecter:
    def __init__(self) :
        self.node=rospy.init_node("detecter")
        self.rgb_subscriber=rospy.Subscriber("",Image,callback=self.Image_callback)
        self.dep_subscriber=rospy.Subscriber("",Image,callback=self.Depth_callback)
        self.dep_publisher=rospy.Publisher("depth",Int8,1)
        self.rgb_image=Image
        self.depth_image=Image
    def Depth_callback(self,msg):
        self.depth_image=msg
        #bridge=CvBridge()
        #self.depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")

    def Image_callback(self,msg):
        self.rgb_image=msg
        #bridge=CvBridge()
        #self.cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    def detecter(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            src = cv2.imread(self.rgb_image)
            qrcoder = cv2.QRCodeDetector()
            points = qrcoder.detect(src)
            #print(points)
            #cv2.drawContours(src, [np.int32(points)], 0, (0, 0, 255), 2)
            depth = (self.depth_image[points(np.array_2d[0,0])][points(np.array_2d[0,1])]+self.depth_image[points(np.array_2d[1,0])][points(np.array_2d[1,1])]+
                     self.depth_image[points(np.array_2d[2,0])][points(np.array_2d[2,1])]+self.depth_image[points(np.array_2d[3,0])][points(np.array_2d[3,1])])/4

            self.dep_publisher.publish(depth)
            rate.sleep()


if  __name__=="__name__":
    c=code_detecter()
    c.detecter()
   
