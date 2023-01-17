import rclpy
# import the ROS2 python libraries
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import ReliabilityPolicy, QoSProfile
import numpy as np



class ReadMap(Node):

    def __init__(self):
        # Here you have the class constructor
        # call super() in the constructor to initialize the Node object
        # the parameter you pass is the node name
        super().__init__('readmap')
        # create the subscriber object
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.output = []

    def map_callback(self, msg):
        self.get_logger().info("Saving map")
        self.get_logger().info(f'height: {msg.info.height}, width: {msg.info.width}, res: {msg.info.resolution}')
        occupancy_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        occupancy_grid = (occupancy_grid == 100).astype(int)
        self.get_logger().info(f'{occupancy_grid.shape}')
        np.save('grid', occupancy_grid)
        # this takes the value at angle 359 (equivalent to angle 0)



def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor
    readmap = ReadMap()
    readmap.get_logger().info("Initializing")
    rclpy.spin(readmap)
    readmap.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()


if __name__ == '__main__':
    main()
