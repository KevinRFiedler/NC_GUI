# -*- coding: utf-8 -*-
"""
RunDemo.py

This file demonstrates how to use the functions in Nanocartography.py to construct a tip/tilt diagram.

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



import Nanocartographer as nc
import matplotlib.pyplot as plt
import InputParameters as ip #Read in adjustable parameters.



#Use the Nanocartographer module to compute the relevant quantities.
print('Constructing Tip/Tilt Diagram')
planes_tip_tilt, poles_tip_tilt, stage_limits = nc.Construct_TipTiltDiagram(ip.a, ip.b, ip.c, ip.alpha_crystal, ip.beta_crystal, ip.gamma_crystal, \
                                                                       ip.alpha_pole, ip.beta_pole, ip.cryst_rot, \
                                                                       ip.sample_rot, ip.horizontal_flip, ip.vertical_flip, \
                                                                       ip.found_pole, ip.is_hexagonal, ip.use_four_index, \
                                                                       ip.alpha_limit, ip.beta_limit, ip.superellipse_param)


#Make the plots and save them.
print('Beginning to plot.')
plt_fig, plt_ax = plt.subplots()

#The order in which these things are plotted determines the legend order and what is over top of what.
stage_limits_sc = nc.Plot_TipTiltCoords(stage_limits, plt_fig, plt_ax, legend_entries='Stage Limits', plt_color='gray')

plane_colors = ['cyan', 'magenta', 'yellow','red']
for plane_ind in range(0, len(planes_tip_tilt)):
    plane = planes_tip_tilt[plane_ind]
    nc.Plot_TipTiltCoords(plane[0], plt_fig, plt_ax, legend_entries=str(plane[1]).replace(']', ')').replace('[', '('), plt_color=plane_colors[plane_ind])
    
pole_colors = ['black', 'red', 'green', 'blue', 'orange', 'pink']
for pole_ind in range(0, len(poles_tip_tilt)):
    pole = poles_tip_tilt[pole_ind]
    nc.Plot_TipTiltCoords(pole[0], plt_fig, plt_ax, legend_entries=pole[1], plt_color=pole_colors[pole_ind], mrk_size=64)

#Make the aspect ratio correct and zoom in on the stage limits.
plt_ax.set_aspect('equal', 'box')
plt.ylim(ip.min_beta, ip.max_beta)
plt.xlim(ip.min_alpha, ip.max_alpha)
plt_ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1)) #Move the legend off the diagram itself.
plt.xlabel('alpha')
plt.ylabel('beta')
plt.style.use('ggplot')


#Save the plots.   
print('Saving plots.')       
plt.savefig('./results/CrystalOrientation.png')


print('Done.')
