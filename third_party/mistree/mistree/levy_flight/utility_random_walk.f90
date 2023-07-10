! "utility_random_walk.f95" generates random walk distributions.

subroutine random_walk_fast_2d(step_size, phi, box_size, x_start, y_start, periodic, length, x, y)
    ! Generates a random walk on a 2D grid.
    !
    ! Parameters
    ! ----------
    ! step_size : float
    !    A set of input step size values.
    ! phi : array
    !    Random angles.
    ! box_size : float
    !    box size parameters.
    ! x_start, y_start : float
    !    Starting x & y coordinates.
    ! periodic : {0, 1}, int
    !    0 = does not enforce periodic boundary conditions.
    !    1 = enforces periodic boundary conditions.
    ! length : int
    !    Length of the x & y coordinates.
    !
    ! Returns
    ! -------
    ! x, y : array
    !    The coordinates of the random walk simulation.

    implicit none

    integer, intent(in) :: length
    real, intent(in) :: step_size(length), phi(length)
    real, intent(in) :: box_size, x_start, y_start
    integer, intent(in) :: periodic
    real, intent(out) :: x(length+1), y(length+1)
    integer :: i
    real :: delta_x, delta_y, x_current, y_current

    x(1) = x_start
    y(1) = y_start

    x_current = x_start
    y_current = y_start

    do i = 2, length+1

        delta_x = step_size(i-1)*cos(phi(i-1))
        delta_y = step_size(i-1)*sin(phi(i-1))

        x_current = x_current + delta_x
        y_current = y_current + delta_y

        if (periodic == 1) then

            do while (x_current < 0. .or. x_current > box_size)
                if (x_current < 0.) then
                    x_current = x_current + box_size
                else if (x_current > box_size) then
                    x_current = x_current - box_size
                end if
            end do

            do while (y_current < 0. .or. y_current > box_size)
                if (y_current < 0.) then
                    y_current = y_current + box_size
                else if (y_current > box_size) then
                    y_current = y_current - box_size
                end if
            end do

        end if

        x(i) = x_current
        y(i) = y_current

    end do

end subroutine random_walk_fast_2d

subroutine random_walk_fast_3d(step_size, phi, theta, box_size, x_start, y_start, z_start, periodic, length, x, y, z)
    ! Generates a random walk on a 3D grid.
    !
    ! Parameters
    ! ----------
    ! step_size : float
    !    A set of input step size values.
    ! phi, theta : array
    !    Random phi (longitude) and theta (latitude) angles.
    ! box_size : float
    !    box size parameters.
    ! x_start, y_start, z_start : float
    !    Starting x & y coordinates.
    ! periodic : {0, 1}, int
    !    0 = does not enforce periodic boundary conditions.
    !    1 = enforces periodic boundary conditions.
    ! length : int
    !    Length of the x & y coordinates.
    !
    ! Returns
    ! -------
    ! x, y, z: array
    !    The coordinates of the random walk simulation.

    implicit none

    integer, intent(in) :: length
    real, intent(in) :: step_size(length), theta(length), phi(length)
    real, intent(in) :: box_size, x_start, y_start, z_start
    integer, intent(in) :: periodic
    real, intent(out) :: x(length+1), y(length+1), z(length+1)
    integer :: i
    real :: delta_x, delta_y, delta_z, x_current, y_current, z_current

    x(1) = x_start
    y(1) = y_start
    z(1) = z_start

    x_current = x_start
    y_current = y_start
    z_current = z_start

    do i = 2, length+1

        delta_x = step_size(i-1)*cos(phi(i-1))*sin(theta(i-1))
        delta_y = step_size(i-1)*sin(phi(i-1))*sin(theta(i-1))
        delta_z = step_size(i-1)*cos(theta(i-1))

        x_current = x_current + delta_x
        y_current = y_current + delta_y
        z_current = z_current + delta_z

        if (periodic == 1) then

            do while (x_current < 0. .or. x_current > box_size)
                if (x_current < 0.) then
                    x_current = x_current + box_size
                else if (x_current > box_size) then
                    x_current = x_current - box_size
                end if
            end do

            do while (y_current < 0. .or. y_current > box_size)
                if (y_current < 0.) then
                    y_current = y_current + box_size
                else if (y_current > box_size) then
                    y_current = y_current - box_size
                end if
            end do

            do while (z_current < 0. .or. z_current > box_size)
                if (z_current < 0.) then
                    z_current = z_current + box_size
                else if (z_current > box_size) then
                    z_current = z_current - box_size
                end if
            end do

        end if

        x(i) = x_current
        y(i) = y_current
        z(i) = z_current

    end do

end subroutine random_walk_fast_3d
