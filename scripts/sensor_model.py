#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from midterm.msg import Obs

def callback (msg):
	global sensor_pub

	sigma = 0.2
	obsmsg = Obs()
	xTrue = msg.pose[1].position.x
	yTrue = msg.pose[1].position.y
	xEst = xTrue + np.random.randn() * sigma
	yEst = yTrue + np.random.randn() * sigma
	obsmsg.z1 = xEst
	obsmsg.z2 = yEst
	sensor_pub.publish(obsmsg)
    

def sensor_model():
	global sensor_pub
    
	rospy.init_node('sensor_model', anonymous=True)
	rospy.Rate(10)
	rospy.Subscriber("/gazebo/model_states",ModelStates,callback)

	sensor_pub = rospy.Publisher('sensor_readings', Obs, queue_size=10)
	rospy.spin()

if __name__ == '__main__':
	sensor_model()


