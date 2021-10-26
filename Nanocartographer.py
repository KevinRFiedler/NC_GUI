# -*- coding: utf-8 -*-
"""
Nanocartographer.py

This utility module contains functions which demonstrate the mathematics contained in the accompanying paper.

List of functions:
    Construct_CubicPoleFamily
    Construct_HexPoleFamilies
    Construct_StageLimits
    Construct_TipTiltDiagram
    Construct_TipTiltSeries_AcrossAlong
    Construct_TipTiltSeries_Cartesian
    Convert_Cartesian2Microscope
    Convert_Microscope2Cartesian
    Convert_Cubic2Native
    Convert_Native2Cubic
    Convert_ThreeIndex2FourIndex
    Convert_FourIndex2ThreeIndex
    Plot_TipTiltCoords
    Rotate_CartesianVector

@author: Kevin Fiedler and Matt Olszta
Date created: 11/6/2020
Date last modified: 10/14/2021

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830

Copyright Battelle Memorial Institute

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


#Universal import statements
import numpy as np



"""
Construct_CubicPoleFamily

This function gives the poles in the family for a cubic system.

Inputs:    
    pole - a 3 component vector that contains the [hkl] of a pole.
     
Outputs:
    unique_poles - an array of all the poles in the pole family.
"""

def Construct_CubicPoleFamily(pole):
    #We want to construct the cyrstallographic family of poles so we have to take all the combinations
    # of the three components as well as their negatives.
    #If we make the pole positive first, then it is easier to see if they are equal.  Not strictly necessary though.
    pole = np.absolute(pole)
    
    #Maybe there is a more elegant way to do this?
    all_poles = [[pole[0], pole[1], pole[2]], \
                 [pole[0], pole[2], pole[1]], \
                 [pole[1], pole[0], pole[2]], \
                 [pole[1], pole[2], pole[0]], \
                 [pole[2], pole[1], pole[0]], \
                 [pole[2], pole[0], pole[1]], \
                 [-pole[0], pole[1], pole[2]], \
                 [-pole[0], pole[2], pole[1]], \
                 [-pole[1], pole[0], pole[2]], \
                 [-pole[1], pole[2], pole[0]], \
                 [-pole[2], pole[1], pole[0]], \
                 [-pole[2], pole[0], pole[1]], \
                 [pole[0], -pole[1], pole[2]], \
                 [pole[0], -pole[2], pole[1]], \
                 [pole[1], -pole[0], pole[2]], \
                 [pole[1], -pole[2], pole[0]], \
                 [pole[2], -pole[1], pole[0]], \
                 [pole[2], -pole[0], pole[1]], \
                 [pole[0], pole[1], -pole[2]], \
                 [pole[0], pole[2], -pole[1]], \
                 [pole[1], pole[0], -pole[2]], \
                 [pole[1], pole[2], -pole[0]], \
                 [pole[2], pole[1], -pole[0]], \
                 [pole[2], pole[0], -pole[1]], \
                 [-pole[0], -pole[1], pole[2]], \
                 [-pole[0], -pole[2], pole[1]], \
                 [-pole[1], -pole[0], pole[2]], \
                 [-pole[1], -pole[2], pole[0]], \
                 [-pole[2], -pole[1], pole[0]], \
                 [-pole[2], -pole[0], pole[1]], \
                 [-pole[0], pole[1], -pole[2]], \
                 [-pole[0], pole[2], -pole[1]], \
                 [-pole[1], pole[0], -pole[2]], \
                 [-pole[1], pole[2], -pole[0]], \
                 [-pole[2], pole[1], -pole[0]], \
                 [-pole[2], pole[0], -pole[1]], \
                 [pole[0], -pole[1], -pole[2]], \
                 [pole[0], -pole[2], -pole[1]], \
                 [pole[1], -pole[0], -pole[2]], \
                 [pole[1], -pole[2], -pole[0]], \
                 [pole[2], -pole[1], -pole[0]], \
                 [pole[2], -pole[0], -pole[1]], \
                 [-pole[0], -pole[1], -pole[2]], \
                 [-pole[0], -pole[2], -pole[1]], \
                 [-pole[1], -pole[0], -pole[2]], \
                 [-pole[1], -pole[2], -pole[0]], \
                 [-pole[2], -pole[1], -pole[0]], \
                 [-pole[2], -pole[0], -pole[1]]
                 ]
         
    #Remove duplicates.
    unique_poles = []
    for test_pole in all_poles:
        add = True
        for pole in unique_poles:
            if pole == test_pole:
                add = False
        if add:
            unique_poles.append(test_pole)
    
    return unique_poles



"""
Construct_HexPoleFamilies

This function returns pole families for hexagonal crystals..

Inputs:    
    N/A
     
Outputs:
    major_pole_families - an array of all the pole families.
"""

def Construct_HexPoleFamilies():
    #This is coded by hand to return the hexagonal pole families because I couldn't figure out how to generate it algorithmically.
    major_pole_families = []
    major_pole_families.append([[0,0,1], [0,0,-1]]) #top and bottom
    major_pole_families.append([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [1,1,0], [-1,-1,0]]) #corners
    major_pole_families.append([[1,-1,0], [-1,1,0], [1,2,0], [-1,-2,0], [2,1,0], [-2,-1,0]]) #middle of faces
    major_pole_families.append([[1,0,-1], [1,0,1], [0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1], [-1,0,1], [-1,0,-1],  [1,1,1], [1,1,-1], [-1,-1,1], [-1,-1,-1]]) #Corners top and bottom
    major_pole_families.append([[1,-1,1], [-1,1,1], [1,2,1], [-1,-2,1], [2,1,1], [-2,-1,1], [1,-1,-1], [-1,1,-1], [1,2,-1], [-1,-2,-1], [2,1,-1], [-2,-1,-1]]) #middle top and bottom
    #major_pole_families.append([[2,-1,0], [-2,1,0], [3,1,0], [-3,-1,0], [1,3,0], [-1,-3,0], [-1,2,0], [1,-2,0], [3,2,0], [-3,-2,0], [2,3,0], [-2,-3,0]])
    
    return major_pole_families


"""
Construct_StageLimits
This function plots the stage limits using a superellipse fitted to Matt's 
microscope data.  The max is 36 in alpha and 32 in beta with an exponent of 3.3.
The parameterized curves are:
    x = a*[cos(t)]^2/r
    y = b*[sin(t)]^2/r

Inputs:    
    alpha_limit - the parameter a in the above formula
    
    beta_limit - the parameter b in the above formula
    
    superellipse_param - the exponent r in the above formula
    
Outputs:
    coords - A list of the tip/tilt coordinates of the stage limits.
"""

def Construct_StageLimits(alpha_limit, beta_limit, superellipse_param):  
    #Superellipse parameters
    a = alpha_limit
    b = beta_limit
    r = superellipse_param
    
    #Find a parameter that goes from 0 to 90, then reflect about x and y to fill curve.
    t_series = np.arange(0, 90, 0.1)
    
    #Turn the series into plottable coordinates.
    x_plot_coords = [[a*np.power(np.cos(np.radians(el)), (2/r))] for el in t_series]
    y_plot_coords = [[b*np.power(np.sin(np.radians(el)), (2/r))] for el in t_series]
    
    #Flip to 2nd quadrant about the y-axis.
    first_flip_x_coords = -np.flip(x_plot_coords)
    first_flip_y_coords = np.flip(y_plot_coords)
    upper_half_x_coords = np.concatenate((x_plot_coords, first_flip_x_coords))
    upper_half_y_coords = np.concatenate((y_plot_coords, first_flip_y_coords))
    
    #Flip to 3rd and 4th quadrants about the x-axis.
    second_flip_x_coords = np.flip(upper_half_x_coords)
    second_flip_y_coords = -np.flip(upper_half_y_coords)
    all_x_coords = np.concatenate((upper_half_x_coords, second_flip_x_coords))
    all_y_coords = np.concatenate((upper_half_y_coords, second_flip_y_coords))
    
    #Plot it in a scatter plot.
    coords = []
    for ind in range(len(all_x_coords)):
        coords.append([float(all_x_coords[ind]), float(all_y_coords[ind])])

    return coords
    
    
"""
Construct_TiltSeries_AcrossAlong
This function returns a tip/tilt series in tip/tilt coordinates at a given angle 
to the x-axis.

Inputs:
    alpha - current tip coordinate of the microscope
    
    beta - current tilt coordinate of the microscope
    
    rotation_angle - angle (in degrees) of the line to the x-axis
    
    tilt_step - size of the steps (in degrees) between entries in the series
    
Outputs:
    tilt_series_across - set of tip/tilt coordinates perpendicular to the given line
    
    tilt_series_along - set of tip/tilt coordinates along a given line at the passed rotation angle
"""

def Construct_TiltSeries_AcrossAlong(alpha, beta, rotation_angle, tilt_step): 
    #For the given angle, construct the Cartesian vectors across and along the interface.
    across_vec = [-np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)), 0]
    along_vec  = [ np.cos(np.radians(rotation_angle)), np.sin(np.radians(rotation_angle)), 0]
    
    #Use our helper function to create the Carteian tip/tilt series for each in the desired steps.
    across_series_Cart = Construct_TipTiltSeries_Cartesian(across_vec, along_vec, tilt_step)
    along_series_Cart = Construct_TipTiltSeries_Cartesian(along_vec, across_vec, tilt_step) 
    
    #Rotate the series to correct location so that alpha/beta are correct.
    #Convert the found pole from tip/tilt to Cartesian then find the cross product.
    beam_axis = [0, 0, 1]
    cart_found_pole = Convert_Microscope2Cartesian([alpha, beta])
    rot_axis = np.cross(beam_axis, cart_found_pole)
    if (beam_axis[0] == cart_found_pole[0]) and (beam_axis[1] == cart_found_pole[1]) and (beam_axis[2] == cart_found_pole[2]):
        #Convert to tip/tilt coordinates
        tilt_series_across = [Convert_Cartesian2Microscope(el) for el in across_series_Cart]
        tilt_series_along = [Convert_Cartesian2Microscope(el) for el in along_series_Cart]
    else:
        rot_axis = rot_axis/np.linalg.norm(rot_axis) #Technically, should be normalized already.
        
        #Compute the angle between the beam and the known pole
        known_angle = np.degrees(np.arccos(np.dot(cart_found_pole, beam_axis)))
        
        #Rotate every vector in the series.
        across_series_Cart_rot = [Rotate_CartesianVector(el, rot_axis, known_angle) for el in across_series_Cart]
        along_series_Cart_rot = [Rotate_CartesianVector(el, rot_axis, known_angle) for el in along_series_Cart]
        
        #Convert to tip/tilt coordinates
        tilt_series_across = [Convert_Cartesian2Microscope(el) for el in across_series_Cart_rot]
        tilt_series_along = [Convert_Cartesian2Microscope(el) for el in along_series_Cart_rot]        
    
    return tilt_series_across, tilt_series_along



"""
Construct_TipTiltDiagram

This function plots the tip/tilt diagram for the given parameters.

Inputs:    
    a - unit cell dimension
    
    b - unit cell dimension
    
    c - unit cell dimension
    
    alpha_crystal - crystallographic angle (in degrees)
    
    beta_crystal - crystallographic angle (in degrees)
    
    gamma_crystal - crystallographic angle (in degrees)
    
    alpha_ref - alpha coordinate of the found pole.
    
    beta_ref - beta coordinate of the found pole.
    
    cryst_rot_angle - rotation of the crystal, in degrees
    
    sample_rot_angle - rotation of the sample/holder, in degrees
    
    horizontal_flip - horizontal loading flip of the holder
    
    vertical_flip - horizontal loading flip of the holder
    
    found_pole - hkl of the found pole in a matrix.
    
    is_hexagonal - logical flag that tells us whether to use the hexagonal grouping for poles and planes
    
    use_four_index - Logical flag that says whether or not to use the 3-index vs 4-index Miller indices 
    
    alpha_limit - max alpha that the stage can reach
    
    beta_limit - max beta that the stage can reach
    
    superellipse_param - exponent of the superellipse that determines roundness/sharpness of corners
    
Outputs:   
    poles - a list of tip/tilt coordinates with legend entries
    
    lines - a list of lines between poles with legend entries
"""

def Construct_TipTiltDiagram(a, b, c, alpha_crystal, beta_crystal, gamma_crystal, alpha_ref, beta_ref, \
                             cryst_rot_angle, sample_rot_angle, horizontal_flip, vertical_flip, found_pole, \
                             is_hexagonal, use_four_index, alpha_limit, beta_limit, superellipse_param):
    ###R_init###
    #Construct the rotation matrix that moves the found pole to correct location, incorporating all the flips.
    beam_axis = [0, 0, 1]
    cubic_found_pole = Convert_Cubic2Native(a, b, c, alpha_crystal, beta_crystal, gamma_crystal, found_pole)
    norm_cubic_found_pole = cubic_found_pole/np.linalg.norm(cubic_found_pole)
    rot_axis = np.cross(beam_axis, norm_cubic_found_pole)
    
    if (beam_axis[0] == norm_cubic_found_pole[0]) and (beam_axis[1] == norm_cubic_found_pole[1]) and (beam_axis[2] == norm_cubic_found_pole[2]):
        #If the beam aligns, then we don't need to move anything yet.
        R_init = np.array([[1, 0, 0], \
                           [0, 1, 0], \
                           [0, 0, 1]])
    else:
        if (beam_axis[0] == norm_cubic_found_pole[0]) and (beam_axis[1] == norm_cubic_found_pole[1]) and (beam_axis[2] == -norm_cubic_found_pole[2]):
            #If the pole is exactly opposite the beam, then we manually set the angle to be pi because the cross product does not return a sensible value.
            r_x = 1
            r_y = 0
            r_z = 0
            known_angle = np.pi   
        else:
            rot_axis_norm = rot_axis/np.linalg.norm(rot_axis) #Technically, should be normalized already.
            r_x = rot_axis_norm[0]
            r_y = rot_axis_norm[1]
            r_z = rot_axis_norm[2]
            #Compute the angle between the beam and the known pole
            known_angle = -np.arccos(np.dot(norm_cubic_found_pole, beam_axis))
        
        R_init = np.array([[r_x**2 + (r_y**2 + r_z**2)*np.cos(known_angle), r_x*r_y*(1-np.cos(known_angle)) - r_z*np.sin(known_angle), r_x*r_z*(1-np.cos(known_angle)) + r_y*np.sin(known_angle)], \
                           [r_x*r_y*(1-np.cos(known_angle)) + r_z*np.sin(known_angle), r_y**2 + (r_x**2 + r_z**2)*np.cos(known_angle), r_y*r_z*(1-np.cos(known_angle)) - r_x*np.sin(known_angle)], \
                           [r_x*r_z*(1-np.cos(known_angle)) - r_y*np.sin(known_angle), r_y*r_z*(1-np.cos(known_angle)) + r_x*np.sin(known_angle), r_z**2 + (r_x**2 + r_y**2)*np.cos(known_angle)]])
    
    ###R_phi###
    R_phi = np.array([[np.cos(np.radians(cryst_rot_angle)), -np.sin(np.radians(cryst_rot_angle)), 0], \
                      [np.sin(np.radians(cryst_rot_angle)),  np.cos(np.radians(cryst_rot_angle)), 0], \
                      [0,                                     0,                                    1]])
    
    ###R_alpha_ref###
    R_alpha_ref = np.array([[1, 0,                              0], \
                            [0, np.cos(np.radians(alpha_ref)), -np.sin(np.radians(alpha_ref))], \
                            [0, np.sin(np.radians(alpha_ref)),  np.cos(np.radians(alpha_ref))]])
 
    ###R_beta_ref###
    R_beta_ref = np.array([[ np.cos(np.radians(beta_ref)),  0, np.sin(np.radians(beta_ref))], \
                           [ 0,                             1, 0], \
                           [-np.sin(np.radians(beta_ref)),  0, np.cos(np.radians(beta_ref))]])
    
    ###R_gamma###
    R_gamma = np.array([[np.cos(np.radians(sample_rot_angle)), -np.sin(np.radians(sample_rot_angle)), 0], \
                        [np.sin(np.radians(sample_rot_angle)),  np.cos(np.radians(sample_rot_angle)), 0], \
                        [0,                                     0,                                    1]])    
    
    ###R_flip_vert###
    if vertical_flip:
        R_flip_vert = np.array([[-1, 0,  0], \
                                [ 0, 1,  0], \
                                [ 0, 0, -1]])
    else:
        R_flip_vert = np.array([[1, 0, 0], \
                                [0, 1, 0], \
                                [0, 0, 1]]) 
    
    ###R_flip_hor###
    if horizontal_flip:
        R_flip_hor = np.array([[1,  0,  0], \
                               [0, -1,  0], \
                               [0,  0, -1]])
    else:
        R_flip_hor = np.array([[1, 0, 0], \
                               [0, 1, 0], \
                               [0, 0, 1]])
    
    ###R_total###
    R_total = np.matmul(R_flip_hor, np.matmul(R_flip_vert, np.matmul(R_gamma, \
                         np.matmul(R_beta_ref, np.matmul(R_alpha_ref, np.matmul(R_phi, R_init))))))
           
    
    #Make the lines between poles
    if is_hexagonal:
        major_plane_families = [[[0,0,1]], \
                                [[0,1,0], [1, 0, 0], [-1,-1,0]], \
                                [[1,-1,0], [1,2,0], [2,1,0]]]
                            
                              
        
        start_vec_family = [[[0,1,0]], \
                            [[0,0,1], [0,0,1], [0,0,1]],\
                            [[0,0,1], [0,0,1], [0,0,1]]]
                            
    else:
        major_plane_families = [[[1,0,0], [0,1,0], [0,0,1]], \
                                [[-1,-1,0], [-1,1,0], [1,0,-1], [1,0,1], [0,-1,1], [0,1,1]], \
                                [[1,1,-1], [1,-1,1], [-1,1,1], [-1,-1,-1], [1,1,1], [1,-1,-1]]]
            
        start_vec_family = [[[0,0,1], [0,0,1], [0,1,0]], \
                            [[0,0,1], [0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]], \
                                [[0,1,1], [0,1,1], [0,-1,1], [0,-1,1], [0,1,-1], [0,1,-1]]]
            
            
    planes_tip_tilt = []
    for ind in range(0, len(major_plane_families)):
        plane_family = major_plane_families[ind]
        start_vecs = start_vec_family[ind]
        family_tip_tilt = []
        for plane_ind in range(0, len(plane_family)):
            start_vecs_native = [Convert_Cubic2Native(a, b, c, alpha_crystal, beta_crystal, gamma_crystal, el) for el in start_vecs]
            plane_family_native = [Convert_Cubic2Native(a, b, c, alpha_crystal, beta_crystal, gamma_crystal, el) for el in plane_family]
            tip_tilt_Cartesian = Construct_TipTiltSeries_Cartesian(start_vecs_native[plane_ind], plane_family_native[plane_ind], 0.5)
            positioned_tip_tilt = [np.matmul(R_total, el) for el in tip_tilt_Cartesian]
            family_tip_tilt += positioned_tip_tilt
            
        planes_tip_tilt.append([[Convert_Cartesian2Microscope(el) for el in family_tip_tilt], plane_family[0]])
        
                 
    #Make the pole families
    major_pole_families = []
    if is_hexagonal:
        major_pole_families = Construct_HexPoleFamilies()
    else:
        major_poles = [[0,0,1], [0,1,1], [1,1,1], [1,1,2], [1,1,4], [1,0,3]] #More can be added if desired. 
        for ind in range(0, len(major_poles)):
            pole = major_poles[ind]
            major_pole_families.append(Construct_CubicPoleFamily(pole))
    
    poles_tip_tilt = []
    for ind in range(0, len(major_pole_families)):
        #Get the pole family, then move it
        pole_family = major_pole_families[ind]
        cubic_pole_family = [Convert_Cubic2Native(a, b, c, alpha_crystal, beta_crystal, gamma_crystal, el) for el in pole_family]
        pole_family_positioned = [np.matmul(R_total, el) for el in cubic_pole_family]
        poles_tip_tilt.append([[Convert_Cartesian2Microscope(el) for el in pole_family_positioned], pole_family[0]])
        
    #If desired, we can convert the planes and poles to 4-index notation.
    if use_four_index:
        for ind in range(0, len(poles_tip_tilt)):
            current_pole = poles_tip_tilt[ind]
            poles_tip_tilt[ind] = [current_pole[0], Convert_ThreeIndex2FourIndex(current_pole[1])]
        for ind in range(0, len(planes_tip_tilt)):
            current_plane = planes_tip_tilt[ind]
            planes_tip_tilt[ind] = [current_plane[0], Convert_ThreeIndex2FourIndex(current_plane[1])]
        
    #Make the stage limits
    stage_limits = Construct_StageLimits(alpha_limit, beta_limit, superellipse_param)

    return planes_tip_tilt, poles_tip_tilt, stage_limits



"""
Construct_TipTiltSeries_Cartesian
This function returns a tip/tilt series in Cartesian coordinates.  Typically it wil
be converted from Cartesian to microscope coordinates.

Inputs:
    init_vec - Cartesian vector which we will rotate.
    
    rot_axis - Axis of rotation that we will rotate init_vec about in Cartesian.
    
    step_size - Size of the rotation steps in degrees.
    
Outputs:
    tip_tilt_series - set of Cartesian 
"""

def Construct_TipTiltSeries_Cartesian(init_vec, rot_axis, step_size):   
    #Start the tip/tilt series with the initial vector.
    current_vec = init_vec;
    tip_tilt_series = [current_vec];
        
    #Figure out the number of rotations that we would need to go through 360 degrees 
    #This is better if it is an even number divisor of 360.
    num_rotations = int(np.ceil(360/step_size))
    for rot_index in range(num_rotations - 1): #This -1 prevents duplicates, but could be changed if 
        current_vec = Rotate_CartesianVector(current_vec, rot_axis, step_size)
        tip_tilt_series.append(current_vec)
        
    return tip_tilt_series



"""
Convert_Cartesian2Microscope
This function takes in a Cartesian vector and gives back the microscope tip/tilt 
coordinates.

Inputs:
    cart_vec - a vector or list of vectors in Cartesian coordinates.
    
Outputs:
    microscope_coords - set of microscope tip/tilt coordinates
"""

def Convert_Cartesian2Microscope(cart_vec):
    #The derivation for this transformation is in the paper, but it boils down to 
    #what rotations would have to occur to get from (0,0,1) to (x,y,z).
    
    #If the vector is not normalized, then we should do so now.
    norm_cart_vec = cart_vec/np.linalg.norm(cart_vec)
    
    alpha = np.degrees(np.arctan2(-norm_cart_vec[1], np.sqrt(norm_cart_vec[0]**2 + norm_cart_vec[2]**2)))
    beta = np.degrees(np.arctan2(norm_cart_vec[0], norm_cart_vec[2]))
    tip_tilt_coords = [alpha, beta]
    
    return tip_tilt_coords



"""
Convert_Microscope2Cartesian
This function takes microscope tip/tilt coordinates and gives back the Cartesian 
vector corresponding to this location.

Inputs:
    microscope_coords - set of microscope tip/tilt coordinates
    
Outputs:
    cart_vec - a vector or list of vectors in Cartesian coordinates.
"""

def Convert_Microscope2Cartesian(tiptilt_vec):
    #The derivation for this transformation is in the paper, but it boils down to 
    #multiplying two rotation matrices.
    
    alpha = np.radians(tiptilt_vec[0])
    beta = np.radians(tiptilt_vec[1])
    
    R_beta = [[np.cos(beta), 0, np.sin(beta)], \
               [0, 1, 0], \
               [-np.sin(beta), 0, np.cos(beta)]]
    R_alpha =  [[1, 0, 0], \
               [0, np.cos(alpha), -np.sin(alpha)], \
               [0, np.sin(alpha), np.cos(alpha)]]
    
    #We start along the beam direction then tip/tilt to the final location.
    start_vec = [0,0,1]

    cart_vec = np.matmul(R_beta, np.matmul(R_alpha, start_vec))
    
    return cart_vec



"""
Convert_Cubic2Native

This function converts a cubic pole into the given crystal system.

Inputs:    
    a - unit cell dimension
    
    b - unit cell dimension
    
    c - unit cell dimension
    
    alpha - crystallographic angle (in degrees)
    
    beta - crystallographic angle (in degrees)
    
    gamma - crystallographic angle (in degrees)
    
    pole - a 3 component vector that contains the [hkl] of a pole.
     
Outputs:
    converted_pole - a pole in the cubic system
"""
    
def Convert_Cubic2Native(a, b, c, alpha, beta, gamma, pole):
    #Convert the angles to radians.
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    #Convert the angles to the intermediary angle delta between the a and b vectors.
    delta = np.arccos((np.cos(gamma) - np.cos(alpha)*np.cos(beta))/(np.sin(alpha)*np.sin(beta)))
    
    #Conversion matrix
    M = np.array([[a*np.sin(beta), b*np.sin(alpha)*np.cos(delta), 0], \
                  [0,              b*np.sin(alpha)*np.sin(delta), 0], \
                  [a*np.cos(beta), b*np.cos(alpha),               c]])
    
    converted_pole = np.matmul(M, pole)
    
    return converted_pole



"""
Convert_Native2Cubic

This function converts a navtive pole into the cubic crystal system.

Inputs:    
    a - unit cell dimension
    
    b - unit cell dimension
    
    c - unit cell dimension
    
    alpha - crystallographic angle (in degrees)
    
    beta - crystallographic angle (in degrees)
    
    gamma - crystallographic angle (in degrees)
    
    pole - a 3 component vector that contains the [hkl] of a pole.
     
Outputs:
    converted_pole - a pole in the native system
"""

def Convert_Native2Cubic(a, b, c, alpha, beta, gamma, pole):
    #Convert the angles to radians.
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    #Convert the angles to the intermediary angle delta between the a and b vectors.
    delta = np.arccos((np.cos(gamma) - np.cos(alpha)*np.cos(beta))/(np.sin(alpha)*np.sin(beta)))
    
    #Conversion matrix
    M_inv = np.array([[1/a*np.sin(beta),              -np.cos(delta)/(a*np.sin(beta)*np.sin(delta)),                                                                         0], \
                      [0,                              1/(b*np.sin(alpha)*np.sin(delta)),                                                                                    0], \
                      [-np.cos(beta)/(c*np.sin(beta)), (np.cos(beta)*np.sin(alpha)*np.cos(delta) - np.sin(beta)*np.cos(alpha))/(c*np.sin(alpha)*np.sin(beta)*np.sin(delta)), 1/c]])
    
    converted_pole = np.matmul(M_inv, pole)
    
    return converted_pole



"""
Convert_ThreeIndex2FourIndex

This function converts a pole from three index Miller notation into four index Miller notation.

Inputs:    
    pole_3 - a pole in three index Miller notation.
     
Outputs:
    pole_4 - a pole in four index Miller notation.
"""

def Convert_ThreeIndex2FourIndex(pole_3):
    #These definitions are just for readability.
    u = pole_3[0]
    v = pole_3[1]
    w = pole_3[2]    
    
    #Convert these to [UVTW]
    #Normally U and V are divided by 3, but this gives us fractions.
    U = 2*u - v
    V = 2*v - u
    T = -(U + V)
    W = 3*w
    
    return([int(U), int(V), int(T), int(W)])
    


"""
Convert_FourIndex2ThreeIndex

This function converts a pole from four index Miller notation into three index Miller notation.

Inputs:    
    pole_4 - a pole in four index Miller notation.
     
Outputs:
    pole_3 - a pole in three index Miller notation.
"""

def Convert_FourIndex2ThreeIndex(pole_4):
    #These definitions are just for readability.
    U = pole_4[0]
    V = pole_4[1]
    T = pole_4[2] #This is redundant information, so it is not actually needed for computation.
    W = pole_4[3]
    
    #Convert to [uvw]
    u = U + 2*V
    v = 2*U + V
    w = W/3

    return([int(u), int(v), int(w)])



"""
Plot_TipTiltCoords
This function plots a set of tip/tilt coords 

Inputs:
    tip_tilt_series - set of tip/tilt coordinates
    
    plt_fig - figure to plot on
    
    plt_ax - axes to plot on
    
    legend_entries - optional argument to provide a legend entry.
    
Outputs:
    sc - the scatter plot that was created
"""

def Plot_TipTiltCoords(tip_tilt_series, plt_fig, plt_ax, legend_entries=None, plt_color='black', mrk_size=9):   
    #Turn the series into plottable coordinates.
    x_plot_coords = [[el[0]] for el in tip_tilt_series]
    y_plot_coords = [[el[1]] for el in tip_tilt_series]
    
    #Plot it in a scatter plot.
    sc = plt_ax.scatter(x_plot_coords, y_plot_coords, s=mrk_size, label=legend_entries, color=plt_color)
    plt_ax.legend()
    plt_fig.canvas.draw_idle()
    
    return sc
  
    
      
"""
Rotate_CartesianVector
This function takes a vector in Cartesian coordinates, rotation axis, and an angle
and applies the rotation matrix required to perform the desired rotation.

Inputs:
    cartesian_vec - vector in Cartesian coordinates
    
    rot_axis - unit vector that defines the rotation axis.
    
    rot_angle - angle of rotation (degrees)
    
Outputs:
    rotated_vec - a vector or list of vectors in Cartesian coordinates.
"""

def Rotate_CartesianVector(cartesian_vec, rot_axis, rot_angle):
    #Check that the rotation axis is normalized.  If it is not, then do it.
    rot_axis_norm = rot_axis/np.linalg.norm(rot_axis)
    
    #Convert to radians so that the trig functions work
    rot_angle = np.radians(rot_angle);
    
    #Construct the rotation matrix from the rotation axis.  The derivation for this 
    #matrix can be found in the appendix of the paper.
    r_x = rot_axis_norm[0]
    r_y = rot_axis_norm[1]
    r_z = rot_axis_norm[2]
    R_rot = np.array([[r_x**2 + (r_y**2 + r_z**2)*np.cos(rot_angle), r_x*r_y*(1-np.cos(rot_angle)) - r_z*np.sin(rot_angle), r_x*r_z*(1-np.cos(rot_angle)) + r_y*np.sin(rot_angle)], \
             [r_x*r_y*(1-np.cos(rot_angle)) + r_z*np.sin(rot_angle), r_y**2 + (r_x**2 + r_z**2)*np.cos(rot_angle), r_y*r_z*(1-np.cos(rot_angle)) - r_x*np.sin(rot_angle)], \
             [r_x*r_z*(1-np.cos(rot_angle)) - r_y*np.sin(rot_angle), r_y*r_z*(1-np.cos(rot_angle)) + r_x*np.sin(rot_angle), r_z**2 + (r_x**2 + r_y**2)*np.cos(rot_angle)]])
    #print(R_rot)
    
    rotated_vec = np.matmul(R_rot, cartesian_vec)
    return rotated_vec