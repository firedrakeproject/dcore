"""
Test the formulae in coord_transforms.py

The strategy is to check that some obvious coordinates go to what we expect.
"""
import numpy as np
from gusto import coord_transforms
import pytest

tol = 1e-12

def test_xy_and_rphi():

    # Consider the following sets of coordinates:
    # (r, phi)     <--> (x, y)
    # (2, 0)       <--> (2, 0)
    # (0.5, pi/2)  <--> (0, 0.5)
    # (10*sqrt(2), -3*pi/4) <--> (-10,-10)
    # (0, 0)       <--> (0, 0)

    rphi_coords = [[2.0, 0.0],
                   [0.5, np.pi/2],
                   [10*np.sqrt(2), -3*np.pi/4],
                   [0.0, 0.0]]

    xy_coords = [[2.0, 0.0],
                 [0.0, 0.5],
                 [-10.0, -10.0],
                 [0.0, 0.0]]

    for i, (rphi, xy) in enumerate(zip(rphi_coords, xy_coords)):
        new_rphi = coord_transforms.rphi_from_xy(xy[0], xy[1])
        new_xy = coord_transforms.xy_from_rphi(rphi[0], rphi[1])

        rphi_correct = ((abs(new_rphi[0] - rphi[0]) < tol)
                        and (abs(new_rphi[1] - rphi[1]) < tol))

        assert rphi_correct, ("""
                              rphi coordinates not correct, got (%.2e %.2e)
                              when expecting (%.2e %.2e)
                              """ % (new_rphi[0], new_rphi[1], rphi[0], rphi[1]))

        xy_correct = ((abs(new_xy[0] - xy[0]) < tol)
                      and (abs(new_xy[1] - xy[1]) < tol))

        assert xy_correct, ("""
                            xy coordinates not correct, got (%.2e %.2e)
                            when expecting (%.2e %.2e)
                            """ % (new_xy[0], new_xy[1], xy[0], xy[1]))


def test_xy_and_rphi_vectors():

    # Consider the following vectors:
    # (r,phi) components  <--> (x,y) components at (x,y) or (r,phi)
    # (2,0)               <--> (2,0)            at (5,0) or (5,0)
    # (0,0.5)             <--> (0,0.5)          at (5,0) or (5,0)
    # (0.5,0)             <--> (0,-0.5)         at (0,-3) or (3,-pi/2)
    # (0,2)               <--> (2,0)            at (0,-3) or (3,-pi/2)

    rphi_coords = [[5.0, 0.0],
                   [5.0, 0.0],
                   [3.0, -np.pi/2],
                   [3.0, -np.pi/2]]

    xy_coords = [[5.0, 0.0],
                 [5.0, 0.0],
                 [0.0, -3.0],
                 [0.0, -3.0]]

    rphi_vectors = [[2.0, 0.0],
                    [0.0, 0.5],
                    [0.5, 0.0],
                    [0.0, 2.0]]

    xy_vectors = [[2.0, 0.0],
                  [0.0, 0.5],
                  [0.0, -0.5],
                  [2.0, 0.0]]

    for i, (rphi, xy, rphi_comp, xy_comp) in enumerate(zip(rphi_coords, xy_coords,
                                                           rphi_vectors, xy_vectors)):

        new_rphi_comp = coord_transforms.rphi_vector_from_xy(xy_comp, xy)
        new_xy_comp = coord_transforms.xy_vector_from_rphi(rphi_comp, rphi)

        rphi_correct = ((abs(new_rphi_comp[0] - rphi_comp[0]) < tol)
                        and (abs(new_rphi_comp[1] - rphi_comp[1]) < tol))

        assert rphi_correct, ("""
                              rphi components not correct, got (%.2e %.2e)
                              when expecting (%.2e %.2e)
                              """ % (new_rphi_comp[0], new_rphi_comp[1],
                                     rphi_comp[0], rphi_comp[1]))

        xy_correct = ((abs(new_xy_comp[0] - xy_comp[0]) < tol)
                      and (abs(new_xy_comp[1] - xy_comp[1]) < tol))

        assert xy_correct, ("""
                            xy components not correct, got (%.2e %.2e)
                            when expecting (%.2e %.2e)
                            """ % (new_xy_comp[0], new_xy_comp[1],
                                   xy_comp[0], xy_comp[1]))


def test_xyz_and_rlonlat():

    # Consider the following sets of coordinates:
    # (r, lon, lat)  <--> (x, y, z)
    # (2,0,pi/2)     <--> (0, 0, 2)
    # (0.5,pi,0)     <--> (-0.5, 0, 0)
    # (10,-pi/2,0)   <--> (0,-10, 0)
    # (0,0,0)        <--> (0, 0, 0)

    rll_coords = [[2.0, 0.0, np.pi/2],
                  [0.5, np.pi, 0.0],
                  [10, -np.pi/2, 0.0],
                  [0.0, 0.0, 0.0]]

    xyz_coords = [[0.0, 0.0, 2.0],
                  [-0.5, 0.0, 0.0],
                  [0.0, -10.0, 0.0],
                  [0.0, 0.0, 0.0]]

    for i, (rll, xyz) in enumerate(zip(rll_coords, xyz_coords)):
        new_rll = coord_transforms.rlonlat_from_xyz(xyz[0], xyz[1], xyz[2])
        new_xyz = coord_transforms.xyz_from_rlonlat(rll[0], rll[1], rll[2])

        rll_correct = ((abs(new_rll[0] - rll[0]) < tol)
                        and (abs(new_rll[1] - rll[1]) < tol)
                        and (abs(new_rll[2] - rll[2]) < tol))

        assert rll_correct, ("""
                             rphi coordinates not correct, got (%.2e %.2e %.2e)
                             when expecting (%.2e %.2e %.2e)
                             """ % (new_rll[0], new_rll[1], new_rll[2],
                                    rll[0], rll[1], rll[2]))

        xyz_correct = ((abs(new_xyz[0] - xyz[0]) < tol)
                       and (abs(new_xyz[1] - xyz[1]) < tol)
                       and (abs(new_xyz[2] - xyz[2]) < tol))

        assert xyz_correct, ("""
                            xyz coordinates not correct, got (%.2e %.2e %.2e)
                            when expecting (%.2e %.2e %.2e)
                            """ % (new_xyz[0], new_xyz[1], new_xyz[2],
                                   xyz[0], xyz[1], xyz[2]))


def test_xyz_and_rlonlat_vectors():

    # Consider the following vectors:
    # (r,lon,lat) components  <--> (x,y,z) components at (x,y,z) or (r,lon,lat)
    # (10,-6,0.5)             <--> (10,-6,0.5)        at (5,0,0) or (5,0,0)
    # (0.7,3,1.2)             <--> (3,-0.7,1.2)       at (0,-0.5,0) or (0.5,-pi/2,0)
    # (2,0,5)                 <--> (5,0,-2)           at (0,0,-15) or (15,0,-pi/2)

    rll_coords = [[5.0, 0.0, 0.0],
                  [0.5, -np.pi/2, 0.0],
                  [15.0, 0.0, -np.pi/2]]

    xyz_coords = [[5.0, 0.0, 0.0],
                  [0.0, -0.5, 0.0],
                  [0.0, 0.0, -15.0]]

    rll_vectors = [[10.0, -6.0, 0.5],
                   [0.7, 3.0, 1.2],
                   [2.0, 0.0, 5.0]]

    xyz_vectors = [[10.0, -6.0, 0.5],
                   [3.0, -0.7, 1.2],
                   [5.0, 0.0, -2.0]]

    for i, (rll, xyz, rll_comp, xyz_comp) in enumerate(zip(rll_coords, xyz_coords,
                                                           rll_vectors, xyz_vectors)):

        new_rll_comp = coord_transforms.rlonlat_vector_from_xyz(xyz_comp, xyz)
        new_xyz_comp = coord_transforms.xyz_vector_from_rlonlat(rll_comp, rll)

        rll_correct = ((abs(new_rll_comp[0] - rll_comp[0]) < tol)
                        and (abs(new_rll_comp[1] - rll_comp[1]) < tol)
                        and (abs(new_rll_comp[2] - rll_comp[2]) < tol))

        assert rll_correct, ("""
                             rlonlat components not correct, got (%.2e %.2e %.2e)
                             when expecting (%.2e %.2e %.2e)
                             """ % (new_rll_comp[0], new_rll_comp[1], new_rll_comp[2],
                                     rll_comp[0], rll_comp[1], rll_comp[2]))

        xyz_correct = ((abs(new_xyz_comp[0] - xyz_comp[0]) < tol)
                       and (abs(new_xyz_comp[1] - xyz_comp[1]) < tol)
                       and (abs(new_xyz_comp[2] - xyz_comp[2]) < tol))

        assert xyz_correct, ("""
                             xyz components not correct, got (%.2e %.2e %.2e)
                             when expecting (%.2e %.2e %.2e)
                             """ % (new_xyz_comp[0], new_xyz_comp[1], new_xyz_comp[2],
                                    xyz_comp[0], xyz_comp[1], xyz_comp[2]))
