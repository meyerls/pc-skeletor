# 'get_mst_class.py' contains the main class function for constructing a minimum
# spanning tree from a given set of points.

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[:-3] + '/coordinates/')

import numpy as np
import coordinate_utility
from . import construct
from . import stats
from . import branches
from . import density as dens
from . import partition


class GetMST:
    """A class for constructing and analysing the minimum spanning tree (MST). Input the
    node positions of a given data set to initiate the class.
    """

    def __init__(self, x=None, y=None, z=None, phi=None, theta=None, ra=None, dec=None,
                 r=None, units='degrees', do_print=False):
        """
        Parameters
        ----------
        x, y, (z) : array
            Cartesian 2D (3D) coordinates.
        phi, theta, (r) : array
            Tomographic (spherical) coordinates.
        ra, dec, (r) : array
            Celestial tomographic (spherical) coordinates.
        units : {'degree', 'radians'}, optional
            The units of the celestial coordinates ra and dec.
        do_print : bool, optional
            Tells the functions whether it is okay for it to print out statements.

        Notes
        -----
        The default of all of the input parameters are set to 'None' such that an internal parameter '_mode'
        of the MST can be set based on the input parameters. Supply:
            * x and y - for 2D cartesian coordinate. "_mode='2D'"
            * x, y and z - for 3D cartesian coordinates.  "_mode='3D'"
            * phi and theta - for tomographic coordinates. "_mode='tomographic'"
            * phi, theta and r - for spherical polar coordinates. "_mode='spherical polar'"
            * ra and dec - for celestial coordinates. "_mode='tomographic celestial'"
            * ra, dec and r - for celestial spherical polar coordinates. "_mode='spherical polar celestial'"
        """
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.theta = theta
        self.ra = ra
        self.dec = dec
        self.r = r
        self.units = units
        self.do_print = do_print
        if self.x is not None and self.y is not None:
            if self.z is None:
                self._mode = '2D'
            else:
                self._mode = '3D'
        elif self.phi is not None and self.theta is not None:
            if self.r is None:
                self._mode = 'tomographic'
            else:
                self._mode = 'spherical polar'
        elif self.ra is not None and self.dec is not None:
            if self.r is None:
                self._mode = 'tomographic celestial'
            else:
                self._mode = 'spherical polar celestial'
        if do_print is True:
            print('MST mode: ', self._mode, ' coordinates')
        self.k_neighbours = 20
        self.edge_length = None
        self.edge_x = None
        self.edge_y = None
        self.edge_z = None
        self.edge_phi = None
        self.edge_theta = None
        self.edge_index = None
        self.degree = None
        self.mean_degree = None
        self.edge_degree = None
        self.branch_index = None
        self.branch_length = None
        self.branch_edge_count = None
        self.branch_shape = None
        self.scale_cut_length = 0.
        self.num_removed_edges_fraction = None

    def define_k_neighbours(self, k_neighbours):
        """Sets the k_neighbours value. This is automatically set to 20 if this is not called.

        Parameters
        ----------
        k_neighbours : int
            The number of nearest neighbours to consider when creating the k-nearest neighbour graph.
        """
        self.k_neighbours = k_neighbours

    def scale_cut(self, scale_cut_length):
        """Defines the scale cut parameters if a scaling cut is required.

        Parameters
        ----------
        scale_cut_length : float
            The minimum allowed length in the k_nearest_neighbour_graph.
        """
        self.scale_cut_length = scale_cut_length

    def construct_mst(self):
        """Constructs the minimum spanning tree from the input data set."""
        if self._mode == '2D':
            if self.scale_cut_length == 0.:
                edge_length, edge_x, edge_y, edge_index = \
                    construct.construct_mst(self.x, self.y, k_neighbours=self.k_neighbours, two_dimensions=True)
            else:
                edge_length, edge_x, edge_y, edge_index, num_removed_edges_fraction = \
                    construct.construct_mst(self.x, self.y, k_neighbours=self.k_neighbours, two_dimensions=True,
                                            scale_cut_length=self.scale_cut_length)
        else:
            if self._mode == 'tomographic':
                x, y, z = \
                    coordinate_utility.spherical_2_unit_sphere(self.phi, self.theta, units=self.units)
                self.x, self.y, self.z = x, y, z
            elif self._mode == 'spherical polar':
                x, y, z = \
                    coordinate_utility.spherical_2_cartesian(self.r, self.phi, self.theta, units=self.units)
                self.x, self.y, self.z = x, y, z
            elif self._mode == 'tomographic celestial':
                phi, theta, x, y, z = \
                    coordinate_utility.celestial_2_unit_sphere(self.ra, self.dec, units=self.units, output='both')
                self.x, self.y, self.z = x, y, z
                self.phi, self.theta = phi, theta
            elif self._mode == 'spherical polar celestial':
                phi, theta, x, y, z = \
                    coordinate_utility.celestial_2_cartesian(self.r, self.ra, self.dec, units=self.units, output='both')
                self.x, self.y, self.z = x, y, z
                self.phi, self.theta = phi, theta
            else:
                pass
            if self.scale_cut_length == 0.:
                edge_length, edge_x, edge_y, edge_z, edge_index = \
                    construct.construct_mst(self.x, self.y, z=self.z, k_neighbours=self.k_neighbours,
                                            two_dimensions=False)
            else:
                edge_length, edge_x, edge_y, edge_z, edge_index, num_removed_edges_fraction = \
                    construct.construct_mst(self.x, self.y, z=self.z, k_neighbours=self.k_neighbours,
                                            two_dimensions=False, scale_cut_length=self.scale_cut_length)
            self.edge_z = edge_z
        self.edge_length, self.edge_x, self.edge_y, self.edge_index = edge_length, edge_x, edge_y, edge_index
        if self.phi is not None:
            self.edge_phi = np.array([self.phi[self.edge_index[0]], self.phi[self.edge_index[1]]])
        if self.theta is not None:
            self.edge_theta = np.array([self.theta[self.edge_index[0]], self.theta[self.edge_index[1]]])
        if self.scale_cut_length != 0.:
            self.num_removed_edges_fraction = num_removed_edges_fraction

    def get_degree(self):
        """Finds the degree of each node in the constructed MST."""
        if self.edge_index is not None:
            self.degree = stats.get_graph_degree(self.edge_index, len(self.x))
        else:
            raise ValueError("'edge_index' are undefined, meaning the minimum spanning tree has yet to be constructed.")

    def get_mean_degree_for_edges(self):
        """Finds the mean degree for each edge.
        """
        if self.degree is not None:
            self.mean_degree = stats.get_mean_degree_for_edges(self.edge_index, self.degree)
        else:
            raise ValueError("The degrees are undefined, meaning they have yet to be calculated.")

    def get_degree_for_edges(self):
        """Gets the degree of the nodes at each end of all edge."""
        if self.degree is not None:
            self.edge_degree = stats.get_degree_for_edges(self.edge_index, self.degree)
        else:
            raise ValueError("The degrees are undefined, meaning they have yet to be calculated.")

    def get_branches(self, box_size=None, sub_divisions=1):
        """Finds the branches of a MST.

        Parameters
        ----------
        box_size : float, optional
            The size of the '2D' or '3D' box.
        sub_divisions : int, optional
            The number of divisions used to divide the data set in each axis. Used for speeding up the branch
            finding algorithm when using many points (> 100000).
        """
        if sub_divisions == 1:
            branch_index, rejected_branch_index = branches.get_branch_index(self.edge_index, self.edge_degree)
        else:
            if self._mode == '2D':
                branch_index, rejected_branch_index = \
                    branches.get_branch_index_sub_divide(sub_divisions, self.edge_index, self.edge_degree,
                                                         box_size=box_size, edge_x=self.edge_x, edge_y=self.edge_y,
                                                         mode='Euclidean', two_dimension=True)
            elif self._mode == '3D':
                branch_index, rejected_branch_index = \
                    branches.get_branch_index_sub_divide(sub_divisions, self.edge_index, self.edge_degree,
                                                         box_size=box_size, edge_x=self.edge_x, edge_y=self.edge_y,
                                                         edge_z=self.edge_z, mode='Euclidean', two_dimension=False)
            else:
                branch_index, rejected_branch_index = \
                    branches.get_branch_index_sub_divide(sub_divisions, self.edge_index, self.edge_degree,
                                                         box_size=None, phi=self.phi, theta=self.theta,
                                                         edge_phi=self.edge_phi, edge_theta=self.edge_theta, mode='spherical')
        self.branch_index = branch_index
        if len(rejected_branch_index) != 0:
            if self.do_print is True:
                print(str(float(len(rejected_branch_index))) + ' branches were incompleted.')
        branch_length = [np.sum(self.edge_length[i]) for i in branch_index]
        self.branch_length = np.array(branch_length)

    def get_branch_edge_count(self):
        """Finds the number of edges included in each branch."""
        branch_edge_count = [float(len(i)) for i in self.branch_index]
        self.branch_edge_count = np.array(branch_edge_count)

    def get_branch_shape(self):
        """Finds the shape of all branches. This is simply the straight line distance between the two ends divided by
        the branch length."""
        self.branch_shape = branches.get_branch_shape(self.edge_index, self.edge_degree, self.branch_index,
                                                      self.branch_length, mode=self._mode, x=self.x, y=self.y, z=self.z)

    def get_stats_vs_density(self, dx, box_size):
        """Computes the relation between the density contrast and the MST statistics.

        Parameters
        ----------
        dx : float
            The length of the individual cells, that the full box will be divided into, across one dimension.
        box_size : float
            The length of the 2D or 3D box across one axis.

        Returns
        -------
        density : array
            Density contrast of each cell.
        mean_degree, mean_edge_length, mean_branch_length, mean_branch_shape : array
            The mean of the 'degree', 'edge length', 'branch length' and 'branch shape' in each respective cell.

        To do
        -----
        Add support for data sets given in 'tomographic' and 'spherical polar' coordinates.
        """
        if self._mode == '2D':
            mean_degree, density = dens.variable_vs_density(self.x, self.y, dx, self.x, self.y, self.degree, box_size,
                                                            mode='2D', get_density=True)
            x_edge_mean = 0.5 * (self.edge_x[0] + self.edge_x[1])
            y_edge_mean = 0.5 * (self.edge_y[0] + self.edge_y[1])
            mean_edge_length = dens.variable_vs_density(self.x, self.y, dx, x_edge_mean, y_edge_mean, self.edge_length,
                                                        box_size, mode='2D')
            branch_index_end = branches.get_branch_end_index(self.edge_index, self.edge_degree, self.branch_index)
            x_branch_mean = 0.5 * (self.x[branch_index_end[0]] + self.x[branch_index_end[1]])
            y_branch_mean = 0.5 * (self.y[branch_index_end[0]] + self.y[branch_index_end[1]])
            mean_branch_length = dens.variable_vs_density(self.x, self.y, dx, x_branch_mean, y_branch_mean,
                                                          self.branch_length, box_size, mode='2D')
            mean_branch_shape = dens.variable_vs_density(self.x, self.y, dx, x_branch_mean, y_branch_mean,
                                                         self.branch_shape, box_size, mode='2D')
            return density, mean_degree, mean_edge_length, mean_branch_length, mean_branch_shape
        elif self._mode == '3D':
            mean_degree, density = dens.variable_vs_density(self.x, self.y, dx, self.x, self.y, self.degree, box_size,
                                                            z=self.z, z_param=self.z, mode='3D', get_density=True)
            x_edge_mean = 0.5 * (self.edge_x[0] + self.edge_x[1])
            y_edge_mean = 0.5 * (self.edge_y[0] + self.edge_y[1])
            z_edge_mean = 0.5 * (self.edge_z[0] + self.edge_z[1])
            mean_edge_length = dens.variable_vs_density(self.x, self.y, dx, x_edge_mean, y_edge_mean, self.edge_length,
                                                        box_size, z=self.z, z_param=z_edge_mean, mode='3D')
            branch_index_end = branches.get_branch_end_index(self.edge_index, self.edge_degree, self.branch_index)
            x_branch_mean = 0.5 * (self.x[branch_index_end[0]] + self.x[branch_index_end[1]])
            y_branch_mean = 0.5 * (self.y[branch_index_end[0]] + self.y[branch_index_end[1]])
            z_branch_mean = 0.5 * (self.z[branch_index_end[0]] + self.z[branch_index_end[1]])
            mean_branch_length = dens.variable_vs_density(self.x, self.y, dx, x_branch_mean, y_branch_mean,
                                                          self.branch_length, box_size, z=self.z, z_param=z_branch_mean,
                                                          mode='3D')
            mean_branch_shape = dens.variable_vs_density(self.x, self.y, dx, x_branch_mean, y_branch_mean,
                                                         self.branch_shape, box_size, z=self.z, z_param=z_branch_mean,
                                                         mode='3D')
            return density, mean_degree, mean_edge_length, mean_branch_length, mean_branch_shape
        else:
            raise ValueError("Computation for 'Spherical polar' or 'tomographic' data sets is currently unsupported.")

    def output_stats(self, include_index=False):
        """Outputs the MST statistics.

        Parameters
        ----------
        include_index : bool, optional
            If true will output the indexes of the nodes for each edge and the indexes of edges in each branch.

        Returns
        -------
        degree : array
            The degree of each node in the MST.
        edge_length : array
            The length of each edge in the MST.
        branch_length : array
            The length of branches in the MST.
        branch_shape : array
            The shape of branches in the MST.
        edge_index : array, optional
            A 2 dimensional array, where the first nested array shows the indexes for the nodes
            on one end of the edge and the second shows the other node.
        branch_index : list, optional
            A list of branches, where each branch is given as a list of the indexes of the member edges.
        """
        if include_index is True:
            return self.degree, self.edge_length, self.branch_length, self.branch_shape, self.edge_index, \
                   self.branch_index
        else:
            return self.degree, self.edge_length, self.branch_length, self.branch_shape

    def _get_stats(self, include_index=False, sub_divisions=1, k_neighbours=None, scale_cut_length=0.):
        """Computes the MST and outputs the statistics.

        Parameters
        ----------
        include_index : bool, optional
            If True will output the indexes of the nodes for each edge and the indexes of edges in each branch.
        sub_divisions : int, optional
            The number of divisions used to divide the data set in each axis. Used for speeding up the branch
            finding algorithm when using many points (> 100000).
        k_neighbours : int, optional
            The number of nearest neighbours to consider when creating the k-nearest neighbour graph.
        scale_cut_length : float, optional
            The minimum allowed length in the k_nearest_neighbour_graph.

        Returns
        -------
        degree : array
            The degree of each node in the MST.
        edge_length : array
            The length of each edge in the MST.
        branch_length : array
            The length of branches in the MST.
        branch_shape : array
            The shape of branches in the MST.
        edge_index : array, optional
            A 2 dimensional array, where the first nested array shows the indexes for the nodes
            on one end of the edge and the second shows the other node.
        branch_index : list, optional
            A list of branches, where each branch is given as a list of the indexes of the member edges.

        Notes
        -----
        This will calculate all the MST statistics by putting the data set through the following functions:
            1) k_neighbours (if k_neighbours is specified)
            2) construct_mst
            3) get_degree
            4) get_degree_for_edges
            5) get_branches
            6) get_branch_shape
            7) output_stats
        """
        if k_neighbours is not None:
            self.define_k_neighbours(k_neighbours)
        if scale_cut_length != 0.:
            self.scale_cut(scale_cut_length=scale_cut_length)
        self.construct_mst()
        self.get_degree()
        self.get_degree_for_edges()
        self.get_branches(sub_divisions=sub_divisions)
        self.get_branch_shape()
        return self.output_stats(include_index=include_index)

    def get_stats(self, include_index=False, sub_divisions=1, k_neighbours=None,
                  scale_cut_length=0., partitions=1):
        """Gets the minimum spanning tree statistics of a partitioned data set. Same inputs as 'get_stats'.

        Parameters
        ----------
        sub_divisions : int, optional
            The number of divisions used to divide the data set in each axis.
        k_neighbours : int, optional
            The number of nearest neighbours to consider when creating the k-nearest neighbour graph.
        scale_cut_length : float, optional
            The minimum allowed length in the k_nearest_neighbour_graph.
        partitions : int
            Number of partitions to divide the data set into.

        Returns
        -------
        degree : array
            The degree of each node in the MST.
        edge_length : array
            The length of each edge in the MST.
        branch_length : array
            The length of branches in the MST.
        branch_shape : array
            The shape of branches in the MST.
        edge_index : list, optional
            A list of 2 dimensional arrays for the nodes in each group.
        branch_index : list, optional
            A list of list of branches, where each branch is given as a list of the indexes of the member edges.
        groups : array, optional
            The assigned groups for each point in the data set (only outputed if include_index=True). Indexes
            here are indexes of the elements in each group.
        """
        if float(partitions) == 1.:
            return self._get_stats(include_index=include_index, sub_divisions=sub_divisions,
                                   k_neighbours=k_neighbours, scale_cut_length=scale_cut_length)
        else:
            # check mode and store all points in temporary array with prefix 'all_'.
            if self._mode == '2D':
                all_x = np.copy(self.x)
                all_y = np.copy(self.y)
                total_sample_size = len(all_x)
            elif self._mode == '3D':
                all_x = np.copy(self.x)
                all_y = np.copy(self.y)
                all_z = np.copy(self.z)
                total_sample_size = len(all_x)
            elif self._mode == 'tomographic':
                all_phi = np.copy(self.phi)
                all_theta = np.copy(self.theta)
                total_sample_size = len(all_phi)
            elif self._mode == 'spherical polar':
                all_phi = np.copy(self.phi)
                all_theta = np.copy(self.theta)
                all_r = np.copy(self.r)
                total_sample_size = len(all_phi)
            elif self._mode == 'tomographic celestial':
                all_ra = np.copy(self.ra)
                all_dec = np.copy(self.dec)
                total_sample_size = len(all_ra)
            elif self._mode == 'spherical polar celestial':
                all_ra = np.copy(self.ra)
                all_dec = np.copy(self.dec)
                all_r = np.copy(self.r)
                total_sample_size = len(all_ra)
            else:
                pass
            # get groupings
            groups = partition.partition_data(total_sample_size, partitions)
            edge_index = []
            branch_index = []
            for i in range(0, partitions):
                group_indices = partition.get_index_for_group(groups, i)
                if self._mode == '2D':
                    self.__init__(x=all_x[group_indices], y=all_y[group_indices])
                elif self._mode == '3D':
                    self.__init__(x=all_x[group_indices], y=all_y[group_indices], z=all_z[group_indices])
                elif self._mode == 'tomographic':
                    self.__init__(phi=all_phi[group_indices], theta=all_theta[group_indices])
                elif self._mode == 'spherical polar':
                    self.__init__(phi=all_phi[group_indices], theta=all_theta[group_indices], r=all_r[group_indices])
                elif self._mode == 'tomographic celestial':
                    self.__init__(ra=all_ra[group_indices], dec=all_dec[group_indices])
                elif self._mode == 'spherical polar celestial':
                    self.__init__(ra=all_ra[group_indices], dec=all_dec[group_indices], r=all_r[group_indices])
                else:
                    pass
                if include_index is False:
                    _degree, _edge_length , _branch_length, _branch_shape = \
                        self._get_stats(sub_divisions=sub_divisions, k_neighbours=k_neighbours, scale_cut_length=scale_cut_length)
                else:
                    _degree, _edge_length , _branch_length, _branch_shape, _edge_index, _branch_index = \
                        self._get_stats(include_index=True, sub_divisions=sub_divisions, k_neighbours=k_neighbours, scale_cut_length=scale_cut_length)
                if i == 0:
                    degree = _degree
                    edge_length = _edge_length
                    branch_length = _branch_length
                    branch_shape = _branch_shape
                else:
                    degree = np.concatenate([degree, _degree])
                    edge_length = np.concatenate([edge_length, _edge_length])
                    branch_length = np.concatenate([branch_length, _branch_length])
                    branch_shape = np.concatenate([branch_shape, _branch_shape])
                if include_index is True:
                    edge_index.append(_edge_index)
                    branch_index.append(_branch_index)
            if include_index is True:
                return degree, edge_length, branch_length, branch_shape, edge_index, branch_index, groups
            else:
                return degree, edge_length, branch_length, branch_shape
            if self._mode == '2D':
                self.x = all_x
                self.y = all_y
            elif self._mode == '3D':
                self.x = all_x
                self.y = all_y
                self.z = all_z
            elif self._mode == 'tomographic':
                self.phi = all_phi
                self.theta = all_theta
            elif self._mode == 'spherical polar':
                self.phi = all_phi
                self.theta = all_theta
                self.r = all_r
            elif self._mode == 'tomographic celestial':
                self.ra = all_ra
                self.dec = all_dec
            elif self._mode == 'spherical polar celestial':
                self.ra = all_ra
                self.dec = all_dec
                self.r = all_r
            else:
                pass

    def clean(self):
        self.x = None
        self.y = None
        self.z = None
        self.ra = None
        self.dec = None
        self.r = None
        self.units = None
        self.do_print = False
        self._mode = None
        self.k_neighbours = 20
        self.phi = None
        self.theta = None
        self.edge_length = None
        self.edge_x = None
        self.edge_y = None
        self.edge_z = None
        self.edge_phi = None
        self.edge_theta = None
        self.edge_index = None
        self.degree = None
        self.mean_degree = None
        self.edge_degree = None
        self.branch_index = None
        self.branch_length = None
        self.branch_edge_count = None
        self.branch_shape = None
        self.scale_cut_length = 0.
        self.num_removed_edges_fraction = None
