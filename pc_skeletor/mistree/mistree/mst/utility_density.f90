! 'utility_density.f90' contains a number of fortran functions for getting the
! average density of points in cells vs the the average value of a certain
! parameter within those cells.

subroutine get_counts_2d(x, y, dx, length, length_of_grid, length_of_points, counts)
    ! Gets the density and average parameter in a cell.
    !
    ! Parameters
    ! ----------
    ! x, y : array
    !    The positions of points.
    ! dx : float
    !    The size of each cell in each axis.
    ! length : int
    !    The length along a single axis of the grid.
    ! length_of_points : int
    !    Length of the x and y coordinates.
    ! length_of_grid : int
    !    Length of the x and y grid.
    !
    ! Returns
    ! -------
    ! counts : array
    !    The number of points in each cell.

    implicit none

    integer, intent(in) :: length, length_of_points, length_of_grid
    real, intent(in) :: x(length_of_points), y(length_of_points), dx
    real, intent(out) :: counts(length_of_grid)

    integer :: i, x_index, y_index, index

    do i = 1, length_of_grid
        counts(i) = 0.
    end do


    do i = 1, length_of_points
        x_index = floor(x(i) / dx)
        y_index = floor(y(i) / dx)
        if (x_index .eq. length) then
            x_index = x_index - 1
        end if
        if (y_index .eq. length) then
            y_index = y_index - 1
        end if
        index = length * x_index + y_index + 1
        counts(index) = counts(index) + 1.
    end do

end subroutine get_counts_2d

subroutine get_counts_3d(x, y, z, dx, length, length_of_grid, length_of_points, counts)
    ! Gets the density and average parameter in a cell.
    !
    ! Parameters
    ! ----------
    ! x, y, z : array
    !    The positions of points.
    ! dx : float
    !    The size of each cell in each axis.
    ! length : int
    !    The length along a single axis of the grid.
    ! length_of_points : int
    !    Length of the x and y coordinates.
    ! length_of_grid : int
    !    Length of the x and y grid.
    !
    ! Returns
    ! -------
    ! counts : array
    !    The number of points in each cell.

    implicit none

    integer, intent(in) :: length, length_of_points, length_of_grid
    real, intent(in) :: x(length_of_points), y(length_of_points), z(length_of_points), dx
    real, intent(out) :: counts(length_of_grid)

    integer :: i, x_index, y_index, z_index, index

    do i = 1, length_of_grid
        counts(i) = 0.
    end do

    do i = 1, length_of_points
        x_index = floor(x(i) / dx)
        y_index = floor(y(i) / dx)
        z_index = floor(z(i) / dx)
        if (x_index .eq. length) then
            x_index = x_index - 1
        end if
        if (y_index .eq. length) then
            y_index = y_index - 1
        end if
        if (z_index .eq. length) then
            z_index = z_index - 1
        end if
        index = (length * length * x_index) + (length * y_index) + z_index + 1
        counts(index) = counts(index) + 1.
    end do

end subroutine get_counts_3d

subroutine get_param_2d(x_param, y_param, param, dx, length, length_of_grid, length_of_param, mean_param)
    ! Gets the density and average parameter in a cell.
    !
    ! Parameters
    ! ----------
    ! x_param, y_param : array
    !    The positions associated with the measured parameter.
    ! param : array
    !    An array of the input parameter.
    ! dx : float
    !    The size of each cell in each axis.
    ! length : int
    !    The length along a single axis of the grid.
    ! length_of_points : int
    !    Length of the x and y coordinates.
    ! length_of_grid : int
    !    Length of the x and y grid.
    !
    ! Returns
    ! -------
    ! mean_param : array
    !    The mean of a parameter in each cell.

    implicit none

    integer, intent(in) :: length, length_of_param, length_of_grid
    real, intent(in) :: x_param(length_of_param), y_param(length_of_param), param(length_of_param), dx
    real, intent(out) :: mean_param(length_of_grid)

    integer :: i, x_index, y_index, index
    real :: counts(length_of_grid), total_param(length_of_grid)

    do i = 1, length_of_grid
        total_param(i) = 0.
        mean_param(i) = 0.
        counts(i) = 0.
    end do

    do i = 1, length_of_param
        x_index = floor(x_param(i) / dx)
        y_index = floor(y_param(i) / dx)
        if (x_index .eq. length) then
            x_index = x_index - 1
        end if
        if (y_index .eq. length) then
            y_index = y_index - 1
        end if
        index = (length * x_index) + y_index + 1
        total_param(index) = total_param(index) + param(i)
        counts(index) = counts(index) + 1.
    end do

    do i = 1, length_of_grid
        mean_param(i) = total_param(i) / counts(i)
    end do

end subroutine get_param_2d

subroutine get_param_3d(x_param, y_param, z_param, param, dx, length, length_of_grid, length_of_param, mean_param)
    ! Gets the density and average parameter in a cell.
    !
    ! Parameters
    ! ----------
    ! x_param, y_param, z_param : array
    !    The positions associated with the measured parameter.
    ! param : array
    !    An array of the input parameter.
    ! dx : float
    !    The size of each cell in each axis.
    ! length : int
    !    The length along a single axis of the grid.
    ! length_of_points : int
    !    Length of the x and y coordinates.
    ! length_of_grid : int
    !    Length of the x and y grid.
    !
    ! Returns
    ! -------
    ! mean_param : array
    !    The mean of a parameter in each cell.

    implicit none

    integer, intent(in) :: length, length_of_param, length_of_grid
    real, intent(in) :: x_param(length_of_param), y_param(length_of_param), z_param(length_of_param), param(length_of_param), dx
    real, intent(out) :: mean_param(length_of_grid)

    integer :: i, x_index, y_index, z_index, index
    real :: counts(length_of_grid), total_param(length_of_grid)

    do i = 1, length_of_grid
        total_param(i) = 0.
        mean_param(i) = 0.
        counts(i) = 0.
    end do

    do i = 1, length_of_param
        x_index = floor(x_param(i) / dx)
        y_index = floor(y_param(i) / dx)
        z_index = floor(z_param(i) / dx)
        if (x_index .eq. length) then
            x_index = x_index - 1
        end if
        if (y_index .eq. length) then
            y_index = y_index - 1
        end if
        if (z_index .eq. length) then
            z_index = z_index - 1
        end if
        index = (length * length * x_index) + (length * y_index) + z_index + 1
        total_param(index) = total_param(index) + param(i)
        counts(index) = counts(index) + 1.
    end do

    do i = 1, length_of_grid
        if (counts(i) > 0.) then
            mean_param(i) = total_param(i) / counts(i)
        else
            mean_param(i) = 0.
        end if
    end do

end subroutine get_param_3d
