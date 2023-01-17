#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

from nav_msgs.msg._odometry import Odometry
from sensor_msgs.msg._imu import Imu
from sensor_msgs.msg._joint_state import JointState

from tf2_ros.buffer import Buffer
from tf2_geometry_msgs.tf2_geometry_msgs import PointStamped
from tf2_ros.transform_listener import TransformListener

from math import sin, cos
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

# imu covariances taken from ros msg
ANG_COV = 4e-8
ACC_COV = 0.0003
# from https://ams.com/documents/20143/36005/AS5601_DS000395_3-00.pdf/9a58e74f-f6d8-53eb-1fa2-ad62d9911ca4?fbclid=IwAR0cpXyVU04m1Yk2PM6_tKzzDxZyeFZv4ZMv0nppTxr2Mz-P6NFXU20xTbA
ENC_COV = 0.043

class KalmanFilter(Node):
    def __init__(self, Tfinal):
        super().__init__('kalman_filter')

        # Time step of analysis
        self.dt = 0.05
        self.Tfinal = Tfinal

        # initialize ROS variables

        self._imu_sub = self.create_subscription(
            Imu, '/imu', self.update_u, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self.update_odom, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self._joints_sub = self.create_subscription(
            JointState, '/joint_states', self.update_enc, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.timer = self.create_timer(self.dt, self.on_timer)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # holds the acceleration and angular velocity
        self.u = np.zeros((2,1))
        # holds the measured state space variables
        self.odom_data = np.zeros((2,1))
        self.enc_data = np.zeros((2,1))

        # holds the last time odom msg was received for transformations
        self.last_time = (0,0)
        
        # initialize KF variables

        ## Prediction estimation - this is your "Priori"
        # xhat represents [x, x_dot, theta, theta_dot] where x is the distance travelled so far and theta is 
        # the angle so far
        # assume the robot starts at 0, verified by inspecting bag data
        self.xhat = np.matrix([0, 0, 0, 0]).transpose() # mean(mu) estimate for the "first" step
        self.P = np.identity(4) # covariance initial estimation 

        # motion model definition - establish your robots "movement"
        # T is thickness in m, r is radius of wheels in m
        T = 0.16
        r = 0.033
        # this is the state space
        self.A = np.matrix([[1, self.dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])
        # inputs are linear acceleration (x direction) and angular velocity (about z)
        self.B = np.matrix([[1 / 2 * self.dt ** 2, self.dt, 0, 0], [0, 0, self.dt, 1]]).transpose()
        # model variance/covariance derivation taken from kalmanfilter.net
        self.R = np.matrix([[ACC_COV*self.dt**2, ACC_COV*self.dt,0,0], 
            [ACC_COV*self.dt,ACC_COV,0,0], [0,0,ANG_COV*self.dt**2, ANG_COV*self.dt], [0,0,ANG_COV*self.dt,ANG_COV]])

        ## Measurement Model - Update component
        # use 2 wheel kinematic model with encoders being [left, right]
        self.C = np.matrix([[0, 1/r, 0, -T/(2*r)], [0, 1/r, 0, T/(2*r)]])

        # this covariance comes from the encoders
        self.Q = ENC_COV * np.identity(2) # this is the sensor model variance-usually characterized to accompany
                                                  # the sensor model already starting

        T_n = np.arange(0, Tfinal, self.dt)
        self.gt = np.zeros([2, len(T_n)])
        self.estimate = np.zeros([2, len(T_n)])
        self.k = 0
    
    # update the imu data
    def update_u(self, imu_msg):
        a_x = imu_msg.linear_acceleration.x
        w_z = imu_msg.angular_velocity.z

        self.u = np.matrix([[a_x, w_z]]).transpose()

    # update the odometry data for comparison since it same as TF
    def update_odom(self, odom_msg):
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y

        self.odom_data = np.matrix([[x, y]]).transpose()
        self.last_time = (odom_msg.header.stamp.sec, odom_msg.header.stamp.nanosec)
    
    # update the encoder data
    def update_enc(self, joint_msg):
        self.enc_data = np.matrix([[joint_msg.velocity[0], joint_msg.velocity[1]]]).transpose()
    
    def on_timer(self):
        # plot the results once we're outside the range
        if self.k >= self.Tfinal / self.dt:
            self.plot_results()
            self.destroy_node()
            return

        # update the ground truth
        self.gt[:,[self.k]] = self.odom_data

        # take a measurement from the encoders
        y = self.enc_data

        #########################################
        ###### Kalman Filter Estimation #########
        #########################################
        # Prediction update
        xhat_k = self.A * self.xhat + self.B * self.u # we do not put noise on our prediction
        P_predict = self.A*self.P*self.A.transpose() + self.R
        # this co-variance is the prediction of essentially how the measurement and sensor model move together
        # in relation to each state and helps scale our kalman gain by giving
        # the ratio. By Definition, P is the variance of the state space, and
        # by applying it to the motion model we're getting a motion uncertainty
        # which can be propogated and applied to the measurement model and
        # expand its uncertainty as well

        # Measurement Update and Kalman Gain
        K = P_predict * self.C.transpose()*np.linalg.inv(self.C*P_predict*self.C.transpose() + self.Q)
        # the pseudo inverse of the measurement model, as it relates to the model covariance
        # if we don't have a measurement for velocity, the P-matrix tells the
        # measurement model how the two should move together (and is normalised
        # in the process with added noise), which is how the kalman gain is
        # created --> detailing "how" the error should be scaled based on the
        # covariance. If you expand P_predict out, it's clearly the
        # relationship and cross-projected relationships, of the states from a
        # measurement and motion model perspective, with a moving scalar to
        # help drive that relationship towards zero (P should stabilise).

        self.xhat = xhat_k + K * (y - self.C * xhat_k)
        self.P = (np.identity(4) - K * self.C) * P_predict # the full derivation for this is kind of complex relying on
                                            # some pretty cool probability knowledge

        # convert to x,y coordinates
        point = PointStamped()
        point.header.frame_id = 'imu_link'
        point.header.stamp.sec = self.last_time[0]
        point.header.stamp.nanosec = self.last_time[1]
        point.point.x = cos(self.xhat[2,0])*self.xhat[0,0]
        point.point.y = sin(self.xhat[2,0])*self.xhat[0,0]

        # transform to base_footprint frame
        point = self.tf_buffer.transform(point, 'base_footprint')
        self.estimate[:,[self.k]] = np.matrix([[point.point.x, point.point.y]]).transpose()

        self.k += 1

    def plot_results(self):
        T = np.arange(0, self.Tfinal, self.dt)

        # calculate mse
        x_mse = np.mean((self.gt[0,:] - self.estimate[0,:])**2)
        y_mse = np.mean((self.gt[1,:] - self.estimate[1,:])**2)
        self.get_logger().info(f'X MSE: {x_mse}, Y_MSE: {y_mse}')

        # plot estimate vs time and ground truth vs time
        plt.figure()
        plt.plot(T, self.estimate[0,:])
        plt.plot(T, self.estimate[1,:])
        plt.plot(T, self.gt[0,:])
        plt.plot(T, self.gt[1,:])
        plt.title('Est. pos vs True pos over time')
        plt.xlabel('time [s]')
        plt.ylabel('distance [m]')
        plt.legend(['x est.', 'y est.', 'x true', 'y true'])

        # plot estimate pos vs true pos
        plt.figure()
        plt.plot(self.estimate[0,:], self.estimate[1,:])
        plt.plot(self.gt[0,:], self.gt[1,:])
        plt.title('Est. pos vs True pos')
        plt.xlabel('x pos [m]')
        plt.ylabel('y pos [m]')
        plt.legend(['pos est.', 'pos true'])
        plt.show()

def main(args=None):
    rclpy.init(args=args)

    print("MTE544 Final Project - Kalman Filter")

    # experimentally determine this is about the length of data
    # makes it easier to deal with numpy matrices
    Tfinal = 22
    kf = KalmanFilter(Tfinal)

    rclpy.spin(kf)
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()