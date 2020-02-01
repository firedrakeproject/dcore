import numpy as np

# This script contains kernels written in Loopy, which are used with par_loops.
# This allows the kernels to be tested individually.

def BoundaryGaussElim(DG1, Vec_DG1):
    """
    A kernel for performing Gaussian elimination for the boundary recovery operation.
    :arg DG1: The scalar DG1 space to be recovered to.
    :arg DG1: The vector DG1 space, representing the coordinates of the scalar space.
    """

    shapes = {"nDOFs": DG1.finat_element.space_dimension(),
              "dim": np.prod(Vec_DG1.shape, dtype=int)}

    elimin_domain = ("{{[i, ii_loop, jj_loop, kk, ll_loop, mm, iii_loop, kkk_loop, iiii]: "
                     "0 <= i < {nDOFs} and 0 <= ii_loop < {nDOFs} and "
                     "0 <= jj_loop < {nDOFs} and 0 <= kk < {nDOFs} and "
                     "0 <= ll_loop < {nDOFs} and 0 <= mm < {nDOFs} and "
                     "0 <= iii_loop < {nDOFs} and 0 <= kkk_loop < {nDOFs} and "
                     "0 <= iiii < {nDOFs}}}").format(**shapes)

    elimin_insts = ("""
                    <int> ii = 0
                    <int> jj = 0
                    <int> ll = 0
                    <int> iii = 0
                    <int> jjj = 0
                    <int> i_max = 0
                    <float64> A_max = 0.0
                    <float64> temp_f = 0.0
                    <float64> temp_A = 0.0
                    <float64> c = 0.0
                    <float64> f[{nDOFs}] = 0.0
                    <float64> a[{nDOFs}] = 0.0
                    <float64> A[{nDOFs},{nDOFs}] = 0.0
                    """
                    # We are aiming to find the vector a that solves A*a = f, for matrix A and vector f.
                    # This is done by performing row operations (swapping and scaling) to obtain A in upper diagonal form.
                    # N.B. several for loops must be executed in numerical order (loopy does not necessarily do this).
                    # For these loops we must manually iterate the index.
                    """
                    if ON_EXT[0] > 0.0
                    """
                    # only do Gaussian elimination for elements with effective coordinates
                    """
                        for i
                    """
                    # fill f with the original field values and A with the effective coordinate values
                    """
                            f[i] = DG1_OLD[i]
                            A[i,0] = 1.0
                            A[i,1] = EFF_COORDS[i,0]
                            if {nDOFs} == 3
                                A[i,2] = EFF_COORDS[i,1]
                            elif {nDOFs} == 4
                                A[i,2] = EFF_COORDS[i,1]
                                A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                            elif {nDOFs} == 6
                    """
                    # N.B we use {nDOFs} - 1 to access the z component in 3D cases
                    # Otherwise loopy tries to search for this component in 2D cases, raising an error
                    """
                                A[i,2] = EFF_COORDS[i,1]
                                A[i,3] = EFF_COORDS[i,{dim}-1]
                                A[i,4] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                                A[i,5] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                            elif {nDOFs} == 8
                                A[i,2] = EFF_COORDS[i,1]
                                A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                                A[i,4] = EFF_COORDS[i,{dim}-1]
                                A[i,5] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                                A[i,6] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                                A[i,7] = EFF_COORDS[i,0]*EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                            end
                        end
                    """
                    # now loop through rows/columns of A
                    """
                        for ii_loop
                            A_max = fabs(A[ii,ii])
                            i_max = ii
                    """
                    # loop to find the largest value in the ii-th column
                    # set i_max as the index of the row with this largest value.
                    """
                            jj = ii + 1
                            for jj_loop
                                if jj < {nDOFs}
                                    if fabs(A[jj,ii]) > A_max
                                        i_max = jj
                                    end
                                    A_max = fmax(A_max, fabs(A[jj,ii]))
                                end
                                jj = jj + 1
                            end
                    """
                    # if the max value in the ith column isn't in the ii-th row, we must swap the rows
                    """
                            if i_max != ii
                    """
                    # swap the elements of f
                    """
                                temp_f = f[ii]  {{id=set_temp_f}}
                                f[ii] = f[i_max]  {{id=set_f_imax, dep=set_temp_f}}
                                f[i_max] = temp_f  {{id=set_f_ii, dep=set_f_imax}}
                    """
                    # swap the elements of A
                    # N.B. kk runs from ii to (nDOFs-1) as elements below diagonal should be 0
                    """
                                for kk
                                    if kk > ii - 1
                                        temp_A = A[ii,kk]  {{id=set_temp_A}}
                                        A[ii, kk] = A[i_max, kk]  {{id=set_A_ii, dep=set_temp_A}}
                                        A[i_max, kk] = temp_A  {{id=set_A_imax, dep=set_A_ii}}
                                    end
                                end
                            end
                    """
                    # scale the rows below the ith row
                    """
                            ll = ii + 1
                            for ll_loop
                                if ll > ii
                                    if ll < {nDOFs}
                    """
                    # find scaling factor
                    """
                                        c = - A[ll,ii] / A[ii,ii]
                                        for mm
                                            A[ll, mm] = A[ll, mm] + c * A[ii,mm]
                                        end
                                        f[ll] = f[ll] + c * f[ii]
                                    end
                                end
                                ll = ll + 1
                            end
                            ii = ii + 1
                        end
                    """
                    # do back substitution of upper diagonal A to obtain a
                    """
                        iii = 0
                        for iii_loop
                    """
                    # jjj starts at the bottom row and works upwards
                    """
                            jjj = {nDOFs} - iii - 1  {{id=assign_jjj}}
                            a[jjj] = f[jjj]   {{id=set_a, dep=assign_jjj}}
                            for kkk_loop
                                if kkk_loop > {nDOFs} - iii_loop - 1
                                    a[jjj] = a[jjj] - A[jjj,kkk_loop] * a[kkk_loop]
                                end
                            end
                            a[jjj] = a[jjj] / A[jjj,jjj]
                            iii = iii + 1
                        end
                    end
                    """
                    # Do final loop to assign output values
                    """
                    for iiii
                    """
                    # Having found a, this gives us the coefficients for the Taylor expansion with the actual coordinates.
                    """
                        if ON_EXT[0] > 0.0
                            if {nDOFs} == 2
                                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0]
                            elif {nDOFs} == 3
                                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1]
                            elif {nDOFs} == 4
                                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]
                            elif {nDOFs} == 6
                                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,{dim}-1] + a[4]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                            elif {nDOFs} == 8
                                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1] + a[4]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[6]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1] + a[7]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                            end
                    """
                    # if element is not external, just use old field values.
                    """
                        else
                            DG1[iiii] = DG1_OLD[iiii]
                        end
                    end
                    """).format(**shapes)

    return (elimin_domain, elimin_insts)
