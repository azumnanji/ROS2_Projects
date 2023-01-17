import numpy as np
from PIL import Image
import math
from sklearn.neighbors import KDTree

X_MAX = 4.797
X_MIN = -0.843
Y_MAX = 5.3
Y_MIN = -3.94

origin = (X_MIN, Y_MAX)


class ParticleFilter:
    ## Initializes particle filter
    # @param n_points      number of particles
    # @param scan_file     numpy file with lidar scans
    # @param map_file      pgm map file
    def __init__(self, n_points, point, scan_file, map_file):
        self.n_points_ = n_points
        self.bin_occupancy_map_ = self.convert_pgm_to_binary_occupancy_map(map_file)
        self.particles_ = self.generate_initial_points(self.n_points_, point)
        self.scans_ = np.load(scan_file)
        self.lidar_standard_deviation_ = 0.2
        self.res_ = 0.03
        self.kdt_ = KDTree(self.get_occupied_points_in_cartesian())
        
    ## Converts pgm file to numpy occupancy map
    # @param map_file      pgm map file
    def convert_pgm_to_binary_occupancy_map(self, map_file):
        map = np.array(Image.open(map_file))

        map = (255 - map) / 255.0

        bin_occupancy_map = np.full_like(map, -1)
        for r in range(len(map)):
            for c in range(len(map[0])):
                if map[r][c] > 0.65:
                    bin_occupancy_map[r][c] = 1
                elif map[r][c] < 0.25:
                    bin_occupancy_map[r][c] = 0

        return bin_occupancy_map

    ## Returns cartesian coordinates from indices corresponding to occupancy map
    # @param i_row      row index
    # @param j_col      col index
    def get_cartesian_from_indices(self, i_row, j_col):
        x_off, y_off = origin

        x_cart = x_off + (j_col)*self.res_
        y_cart = y_off - (i_row)*self.res_

        return [x_cart, y_cart]

    ## Returns list of cartesian points of occupied cells
    def get_occupied_points_in_cartesian(self):
        pts = []

        for i in range(self.bin_occupancy_map_.shape[0]):
            for j in range(self.bin_occupancy_map_.shape[1]):
                if self.bin_occupancy_map_[i][j] == 1:
                    pts.append(self.get_cartesian_from_indices(i,j))
        return pts

    ## Generates a specified number of given particles through uniform distribution
    # @param n_points      number of particles
    def generate_initial_points(self, n_points, point):
        x = np.random.uniform(*point[0], n_points)
        y = np.random.uniform(*point[1], n_points)
        phi = np.random.uniform(0, 2*math.pi, n_points)
        points = np.column_stack((x, y, phi))
        return points

    ## Transforms a lidar scan to reference frame of given particle
    # @param scan      lidar scan
    # @param particle  particle point
    def transform_to_map_frame(self, scan, particle):
        T = self.get_transform(particle)

        # multiply transformation matrix by every cartesian point in scan
        transformed_scan = scan @ T.T
        # delete the extra columns so we only leave [x,y]
        transformed_scan = np.delete(transformed_scan, 3, 1)
        transformed_scan = np.delete(transformed_scan, 2, 1)
        return transformed_scan

    ## Returns the transformation matrix for a particle
    # @param particle   particle point
    def get_transform(self, particle):
        T = np.array([[np.cos(particle[2]), -np.sin(particle[2]), 0, particle[0]],
        [np.sin(particle[2]), np.cos(particle[2]), 0, particle[1]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

        return T
    
    def calculate_metrics(self, scan):
        average = np.average(self.particles_, axis=0)
        variance = np.var(self.particles_, axis=0)
        clean_scan = scan[~np.isnan(scan).any(axis=1)]
        clean_scan = clean_scan[~np.isinf(clean_scan).any(axis=1)]
        transformed_scan = self.transform_to_map_frame(clean_scan, average)
        distances = self.kdt_.query(transformed_scan, k=1)[0][:]
        mse = np.average(distances**2)
        print("average: ", average)
        print("variance: ", variance)
        print("mse: ", mse)

    
    ## Run the particle filter algorithm
    def run(self):
        print("Running particle filter with", len(self.scans_), "scans and", self.particles_.shape[0], "particles")
        i = 1

        for scan in self.scans_:
            if np.var(self.particles_, axis=0)[0] < 1e-5:
                break
            print("Scan", i)
            # remove all rows with nan and inf values from the scan
            clean_scan = scan[~np.isnan(scan).any(axis=1)]
            clean_scan = clean_scan[~np.isinf(clean_scan).any(axis=1)]
            weights = []
            for particle in self.particles_:
                transformed_scan = self.transform_to_map_frame(clean_scan, particle)

                distances = self.kdt_.query(transformed_scan, k=1)[0][:]

                weight = np.sum(np.exp(-(distances**2)/(2*self.lidar_standard_deviation_**2)))
                weights.append(weight)

            weights = np.array(weights)
            norm_weights = weights / np.sum(weights)    #normalize so that all particles combined add to 1

            choice = np.random.choice(len(self.particles_), len(self.particles_), p=norm_weights)
            self.particles_ = self.particles_[choice]

            i += 1

        # calculate average, variance, mse and accuracy
        self.calculate_metrics(self.scans_[-1])

        

point5 = [(4, 4.75), (-3.6, -2.6)]
point7 = [(2.5, 3.6), (-3.5, -2.4)]
particle_filter = ParticleFilter(5000, point7, scan_file='point7.npy', map_file='map_maze_2.pgm')

particle_filter.run()


            


