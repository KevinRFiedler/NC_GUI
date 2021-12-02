# -*- coding: utf-8 -*-
from flask import render_template, redirect, url_for
from app import app
from app.forms import ParameterForm
import io
import base64
import os
import InputParameters
import Nanocartographer as nc
import InputParameters as ip #Read in adjustable parameters.
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    full_filename = os.path.join('static', 'TipTiltMaps', 'CrystalOrientation.png')
    form = ParameterForm()
    
    #if form.validate_on_submit():#Use this if we care about checking inputs with DataValidators in the form.
    # db logic goes here
    #return render_template('test.html')
    
    alpha_pole = float(form.alpha.data)
    beta_pole = float(form.beta.data)
    
    planes_tip_tilt, poles_tip_tilt, stage_limits = nc.Construct_TipTiltDiagram(ip.a, ip.b, ip.c, ip.alpha_crystal, ip.beta_crystal, ip.gamma_crystal, \
                                                                           alpha_pole, beta_pole, ip.cryst_rot, \
                                                                           ip.sample_rot, ip.horizontal_flip, ip.vertical_flip, \
                                                                           ip.found_pole, ip.is_hexagonal, ip.use_four_index, \
                                                                           ip.alpha_limit, ip.beta_limit, ip.superellipse_param)
    
    plt_fig = Figure()    
    plt_ax = plt_fig.subplots()

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
    plt_ax.set_ylim(ip.min_beta, ip.max_beta)
    plt_ax.set_xlim(ip.min_alpha, ip.max_alpha)
    plt_ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1)) #Move the legend off the diagram itself.
    plt_ax.set_xlabel('alpha')
    plt_ax.set_ylabel('beta')
    plt_ax.grid(visible=True)
    #plt_ax.set_style.use('ggplot')
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(plt_fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')


    
    return render_template('index.html', image = pngImageB64String, form = form)
