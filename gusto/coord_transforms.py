"""
Some common coordinate transforms.
"""
from firedrake import sin, cos, sqrt, Max, Min, asin, atan_2

__all__ = ["xyz_from_rlonlat", "rlonlat_from_xyz", "xy_from_rphi", "rphi_from_xy",
           "xyz_vector_from_rlonlat", "rlonlat_vector_from_xyz", "xy_vector_from_rphi",
           "rphi_vector_from_xy"]


def xyz_from_rlonlat(r, lon, lat):
    """
    Returns the global Cartesian coordinates x, y, z from
    spherical r, lon and lat coordinates.

    Result is returned in metres.

    :arg r: radius in metres.
    :arg lon: longitude in radians.
    :arg lat: latitude in radians.
    """

    x = r * cos(lon) * cos(lat)
    y = r * sin(lon) * cos(lat)
    z = r * sin(lat)

    return x, y, z


def rlonlat_from_xyz(x, y, z):
    """
    Returns the spherical r, lon and lat coordinates from
    the global Cartesian x, y, z coordinates.

    Result is returned in metres and radians.

    :arg x: x-coordinate in metres.
    :arg y: y-coordinate in metres.
    :arg z: z-coordinate in metres.
    """

    lon = atan_2(y, x)
    r = sqrt(x**2 + y**2 + z**2)
    l = sqrt(x**2 + y**2)
    lat = atan_2(z, l)

    return r, lon, lat


def xy_from_rphi(r, phi):
    """
    Returns the global Cartesian x, y coordinates from
    the plane-polar r and phi coordinates.

    Result is returned in metres.

    :arg r: radius in metres.
    :arg phi: angle from x-axis in radians.
    """

    x = r * cos(phi)
    y = r * sin(phi)

    return x, y


def rphi_from_xy(x, y):
    """
    Returns the plane-polar coordinates r and phi from
    the global Cartesian x and y coordinates.

    Result is returned in metres and radians.

    :arg x: x-coordinate in metres.
    :arg y: y-coordinate in metres.
    """

    r = sqrt(x**2 + y**2)
    phi = atan_2(y, x)

    return r, phi


def xyz_vector_from_rlonlat(rlonlat_vector, position_vector):
    """
    Returns the Cartesian x, y and z components of a vector from a
    vector whose components are in r, lon and lat spherical coordinates.
    Needs a position vector, whose components are also assumed to be in
    spherical coordinates.

    :arg rlonlat_vector: a vector whose components are spherical lon-lat
    components.
    :arg position_vector: the position vector in spherical lon-lat coordinates,
    i.e. the radius (in metres), longitude and latitude (in radians).
    """

    r = position_vector[0]
    lon = position_vector[1]
    lat = position_vector[2]

    xyz_vector = [0.0, 0.0, 0.0]

    # f is our vector, e_i is the ith unit basis vector
    # f = f_r * e_r + f_lon * e_lon + f_lat * e_lat
    # We want f = f_x * e_x + f_y * e_y + f_z * e_z

    # f_x = dot(f, e_x)
    # e_x = cos(lon)*cos(lat) * e_r - sin(lon) * e_lon - cos(lon)*sin(lat) * e_lat
    xyz_vector[0] = (cos(lon)*cos(lat) * rlonlat_vector[0]
                     - sin(lon) * rlonlat_vector[1]
                     - cos(lon)*sin(lat) * rlonlat_vector[2])

    # f_y = dot(f, e_y)
    # e_y = sin(lon)*cos(lat) * e_r + cos(lon) * e_lon - sin(lon)*sin(lat) * e_lat
    xyz_vector[1] = (sin(lon)*cos(lat) * rlonlat_vector[0]
                     + cos(lon) * rlonlat_vector[1]
                     - sin(lon)*sin(lat) * rlonlat_vector[2])

    # f_z = dot(f, e_z)
    # e_z = sin(lat) * e_r + cos(lat) * e_lat
    xyz_vector[2] = (sin(lat) * rlonlat_vector[0]
                     + cos(lat) * rlonlat_vector[2])


    return xyz_vector


def rlonlat_vector_from_xyz(xyz_vector, position_vector):
    """
    Returns the spherical r, lon and lat components of a vector from a
    vector whose components are in x, y, z Cartesian coordinates.
    Needs a position vector, whose components are also assumed to be in
    Cartesian coordinates.

    :arg xyz_vector: a vector whose components are the Cartesian x, y and z
    components.
    :arg position_vector: the position vector in Cartesian x, y and z components,
    i.e. the x, y and z values of the position (in metres)
    """

    x = position_vector[0]
    y = position_vector[1]
    z = position_vector[2]

    r, lon, lat = rlonlat_from_xyz(x, y, z)

    rlonlat_vector = [0.0, 0.0, 0.0]

    # f is our vector, e_i is the ith unit basis vector
    # f = f_x * e_x + f_y * e_y + f_z * e_z
    # We want f = f_r * e_r + f_lon * e_lon + f_lat * e_lat

    # f_r = dot(f, e_r)
    # e_r = x/r * e_x + y/r * e_y + z/r * e_z
    rlonlat_vector[0] = (cos(lon) * cos(lat) * xyz_vector[0] +
                         sin(lon) * cos(lat) * xyz_vector[1] +
                         sin(lat) * xyz_vector[2])

    # f_lon = dot(f, e_lon)
    # e_lon = -y/l * e_x + x/l * e_y
    rlonlat_vector[1] = (-sin(lon) * xyz_vector[0]
                         + cos(lon) * xyz_vector[1])

    # f_lat = dot(f, e_lat)
    # e_lat = -x*z/(r*l) * e_x - y*z/(r*l) * e_y + l/r * e_z
    rlonlat_vector[2] = (-cos(lon) * sin(lat) * xyz_vector[0]
                         -sin(lon) * sin(lat) * xyz_vector[1]
                         + cos(lat) * xyz_vector[2])

    return rlonlat_vector


def xy_vector_from_rphi(rphi_vector, position_vector):
    """
    Returns the Cartesian x and y components of a vector from a
    vector whose components are in r, phi plane-polar coordinates.
    Needs a position vector, whose components are also assumed to be in
    plane-polar coordinates.

    :arg rphi_vector: a vector whose components are plane-polar r-phi components.
    :arg position_vector: the position vector in plane-polar r-phi coordinates,
    i.e. the radius (in metres), phi (in radians).
    """

    r = position_vector[0]
    phi = position_vector[1]

    xy_vector = [0.0, 0.0]

    # f is our vector, e_i is the ith unit basis vector
    # f = f_r * e_r + f_phi * e_phi
    # We want f = f_x * e_x + f_y * e_y

    # f_x = dot(f, e_x)
    # e_x = cos(phi) * e_r - sin(phi) * e_phi
    xy_vector[0] = cos(phi) * rphi_vector[0] - sin(phi) * rphi_vector[1]

    # f_y = dot(f, e_y)
    # e_y = sin(phi) * e_r + cos(phi) * e_phi
    xy_vector[1] = sin(phi) * rphi_vector[0] + cos(phi) * rphi_vector[1]

    return xy_vector


def rphi_vector_from_xy(xy_vector, position_vector):
    """
    Returns the plane-polar r and phi components of a vector from a
    vector whose components are in x, y Cartesian coordinates.
    Needs a position vector, whose components are also assumed to be in
    Cartesian coordinates.

    :arg xy_vector: a vector whose components are the Cartesian x and y
    components.
    :arg position_vector: the position vector in Cartesian x and y components,
    i.e. the x and y values of the position (in metres)
    """

    x = position_vector[0]
    y = position_vector[1]

    phi = atan_2(y, x)

    rphi_vector = [0.0, 0.0]

    # f is our vector, e_i is the ith unit basis vector
    # f = f_x * e_x + f_y * e_y
    # We want f = f_r * e_r + f_phi * e_phi

    # f_r = dot(f, e_r)
    # e_r = x/r * e_x + y/r * e_y
    rphi_vector[0] = cos(phi) * xy_vector[0] + sin(phi) * xy_vector[1]

    # f_phi = dot(f, e_phi)
    # e_phi = -y/r * e_x + x/r * e_y
    rphi_vector[1] = - sin(phi) * xy_vector[0] + cos(phi) * xy_vector[1]

    return rphi_vector
