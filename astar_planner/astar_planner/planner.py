import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

from action_files.action import Planner

from nav_msgs.msg._occupancy_grid import OccupancyGrid
from nav_msgs.msg._path import Path

from geometry_msgs.msg._pose import Pose
from geometry_msgs.msg._pose_stamped import PoseStamped

from .a_star import astar

import numpy as np


class PlannerActionServer(Node):

    def __init__(self):
        super().__init__('planner_action_server')
        self.maze = None
        
        self._grid_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.create_maze, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self._path_pub = self.create_publisher(Path, '/calc_path', 10)
        self._action_server = ActionServer(
            self,
            Planner,
            'planner',
            self.execute_callback)

        self.cost_threshold = 70
        self.path_in_poses = []
        self.ready = False
        self.res_ = None
        self.origin_ = None
        self.get_logger().info('Initialized Class')
    
    def create_maze(self, msg):
        if not self.ready and len(msg.data) > 0 and msg.info.height != 0 and msg.info.width != 0:
            self.maze = np.flip(np.array(msg.data).reshape((msg.info.height, msg.info.width)), axis=0)
            self.res_ = msg.info.resolution
            self.origin_ = (msg.info.origin.position.x, msg.info.origin.position.y + msg.info.height*self.res_)

            self.get_logger().info('Initialized map')
            self.get_logger().info(f'Origin: {self.origin_}')
            self.ready = True

    def get_indices_from_pose(self, pose, target_point):
        off_x, off_y = self.origin_

        col_j = round((pose.position.x - off_x) / self.res_)
        row_i = round((off_y - pose.position.y) / self.res_)
        self.get_logger().info(f'{target_point} position: {pose.position.x}, {pose.position.y}')
        self.get_logger().info(f'{target_point} index: {row_i}, {col_j}')
        self.get_logger().info(f'{target_point} cost: {self.maze[row_i][col_j]}')

        return (row_i, col_j)
    
    # do this in astar
    def get_pose_from_indices(self, path_in_indices):
        result = Planner.Result()
        path = Path()
        path.header.frame_id = '/map'

        off_x, off_y = self.origin_
        for (row, col) in path_in_indices:
            p_s = PoseStamped()
            pose = Pose()
            pose.position.x = col * self.res_ + off_x
            pose.position.y = off_y - row * self.res_ 
            result.path.append(pose)
            p_s.pose = pose
            path.poses.append(p_s)
        
        self._path_pub.publish(path)
        return result
        

    def execute_callback(self, goal_handle):
        result = Planner.Result()
        if not self.ready:
            self.get_logger().warn('Failed to initialize map. Have you started the map server?')
            goal_handle.abort()
            return result
        
        self.get_logger().info('Executing goal...')
        start_indices = self.get_indices_from_pose(goal_handle.request.start_position, 'Start')
        goal_indices = self.get_indices_from_pose(goal_handle.request.goal_position, 'Goal')
        try:
            path_in_indices = astar(self.maze, start_indices, goal_indices, self.cost_threshold)
        except AssertionError as e:
            print(e)
            goal_handle.abort()
            return result
            
        if not path_in_indices:
            self.get_logger().warn('Failed to find valid path!')
            goal_handle.abort()
            return result
        
        self.get_logger().info('Completed astar...')
        result = self.get_pose_from_indices(path_in_indices)

        self.get_logger().info('Returning Result')
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)

    planner_action_server = PlannerActionServer()

    rclpy.spin(planner_action_server)


if __name__ == '__main__':
    main()