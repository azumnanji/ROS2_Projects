from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    return LaunchDescription([
      ExecuteProcess(
      	cmd = ['ros2', 'bag', 'play',  '/home/nanji/ros2_ws/src/kalman_filter/path'],
      ),
      Node(
        package='kalman_filter',
        executable='kalman_filter',
        output = 'screen'),
      
    ])

# TimerAction(
#         period=3.0,
#         actions=[
#             Node(
#                 package='kalman_filter',
#                 executable='kalman_filter',
#                 output = 'screen'),
#         ]),