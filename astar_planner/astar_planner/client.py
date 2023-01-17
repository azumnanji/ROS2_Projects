import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

from action_files.action import Planner
from action_msgs.msg import GoalStatus

from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer

from geometry_msgs.msg._twist import Twist
from geometry_msgs.msg._pose import Pose
from scipy.spatial.transform import Rotation
import numpy as np

import sys
import math

LOCAL_FRAME = 'base_footprint'
ODOM_FRAME = 'odom'
MAP_FRAME = 'map'


class PlannerActionClient(Node):

    def __init__(self):
        super().__init__('planner_action_client')
        self._action_client = ActionClient(self, Planner, 'planner')
        # subscriber to tf for pose
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.start_pos = None
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.path_in_poses = Planner.Result()
        self.timer = self.create_timer(0.05, self.on_timer)
        self.index_in_path = 0
        self.thres = 0.1
        self.path_ready = False
        self.client_ready = False


    def send_goal(self, goal_pose):
        while not self.client_ready:
            try:
                self.start_pos = self.tf_buffer.lookup_transform(
                                MAP_FRAME,
                                LOCAL_FRAME,
                                rclpy.time.Time())
                self.client_ready = True
            except TransformException as e:
                self.get_logger().info(
                        f'Could not transform {LOCAL_FRAME} to {MAP_FRAME}',once=True)
            rclpy.spin_once(self)
        
        goal_msg = Planner.Goal()
        goal_msg.start_position = Pose()
        self.get_logger().info(f'Start: {self.start_pos.transform.translation.x}, {self.start_pos.transform.translation.y}')
        goal_msg.start_position.position.x = self.start_pos.transform.translation.x
        goal_msg.start_position.position.y = self.start_pos.transform.translation.y
        self.get_logger().info(f'Goal: {goal_pose.position.x}, {goal_pose.position.y}')
        goal_msg.goal_position = goal_pose

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal failed with status: {0}'.format(status))
            return
        
        debug_path = [(p.position.x, p.position.y) for p in result.path]

        self.get_logger().info(f'Goal succeeded!')
        self.path_in_poses = result.path
        self.path_ready = True


    def on_timer(self):
        # get tf
        # send vel commands to robot
        if self.path_ready:
            try:
                t = self.tf_buffer.lookup_transform(MAP_FRAME, LOCAL_FRAME, rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform {LOCAL_FRAME} to {MAP_FRAME}: {ex}')
                return
            delta_x = self.path_in_poses[self.index_in_path].position.x - t.transform.translation.x
            delta_y = self.path_in_poses[self.index_in_path].position.y - t.transform.translation.y
            if abs(delta_x) <= self.thres and abs(delta_y) <= self.thres:
                if self.index_in_path == len(self.path_in_poses) - 1:
                    self.get_logger().info(f"Robot is at goal position! Terminate")
                    msg = Twist()
                    self.publisher.publish(msg)
                    rclpy.shutdown()
                else:
                    self.index_in_path += 1
            else:
                msg = Twist()
                scale_rotation_rate = 0.25
                # msg.angular.z = scale_rotation_rate 
                angle = math.atan2(
                    delta_y,
                    delta_x)
                
                heading = Rotation.from_quat([-t.transform.rotation.x, -t.transform.rotation.y, -t.transform.rotation.z, t.transform.rotation.w]).as_matrix()[0][:2]

                delta = (delta_x, delta_y)
                delta_mag = math.sqrt(delta[0]**2 + delta[1]**2)
                delta_unit = (delta[0]/delta_mag, delta[1]/delta_mag) # desired heading

                # scales from -1 to 1 depending on direction of rotation and how fast should rotate
                rotation = np.sign(np.cross(heading, delta_unit))*math.acos(np.dot(heading, delta_unit))
                
                # if facing exactly 180 deg away it will think it doesn't need to rotate so manually make sure that doesn't happen
                if rotation == 0 and np.cross(heading, delta_unit) < 0:
                    rotation = 1
                
                twist = Twist()

                if abs(rotation) < 0.1:
                    twist.linear.x = 0.1
                twist.angular.z = rotation*0.5
                
                
                # if abs(angle) > 0.1:

                # if msg.angular.z > math.pi/2:
                #     msg.angular.z = math.pi/2
                # if msg.angular.z < -math.pi/2:
                #     msg.angular.z = -math.pi/2

                # distance = math.sqrt(
                #     delta_x ** 2 +
                #     delta_y ** 2)
                # scale_forward_speed = 0
                
                # if distance < 1:
                #     scale_forward_speed = 0.1
                # else:
                #     scale_forward_speed = 0.5
                # msg.linear.x = scale_forward_speed * distance
                
                self.publisher.publish(twist)
        

def main(args=None):
    if len(sys.argv) < 3:
        print("usage: client arg1 arg2")
    else:
        rclpy.init(args=args)

        action_client = PlannerActionClient()
        # wait for server to start

        goal_pose = Pose()
        goal_pose.position.x = float(sys.argv[1])
        goal_pose.position.y = float(sys.argv[2])

        future = action_client.send_goal(goal_pose)

        rclpy.spin(action_client)


if __name__ == '__main__':
    main()