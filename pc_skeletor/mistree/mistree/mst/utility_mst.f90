! 'utility_mst.f90' contains a fortran functions for getting the degree from the
! edge indices of a minimum spanning tree.

subroutine get_degree_for_index(index1, index2, number_of_nodes, number_of_edges, degree)
    ! Given the edge index this will compute the degree of each node.
    !
    ! Parameters
    ! ----------
    ! index1, index2 : array
    !    The index of the edges of a tree, where the '1' and '2' refer to the ends of each edge.
    ! number_of_nodes : integer
    !    The array integer length of the nodes used to construct the tree.
    ! number_of_edges : integer
    !    The array integer length of the edges forming the constructed tree.
    !
    ! Returns
    ! -------
    ! degree : array
    !    The degree for each node, i.e. the number of edges attached to each node.ß

    implicit none

    integer, intent(in) :: number_of_nodes, number_of_edges
    integer, intent(in) :: index1(number_of_edges), index2(number_of_edges)
    double precision, intent(out) :: degree(number_of_nodes)

    integer :: i

    do i = 1, number_of_nodes
        degree(i) = 0.
    end do

    do i = 1, number_of_edges
        degree(index1(i)+1) = degree(index1(i)+1) + 1.
        degree(index2(i)+1) = degree(index2(i)+1) + 1.
    end do

end subroutine get_degree_for_index
