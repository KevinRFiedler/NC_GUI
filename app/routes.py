# -*- coding: utf-8 -*-
from flask import render_template
from app import app
from app.forms import ParameterForm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
import Nanocartographer as nc



@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    #Use a form to determine the parameters that will make up the tip/tilt map.
    form = ParameterForm()
    
    #Parameters of the crystal system for the sample (cubic, trigonal, etc.)
    a = float(form.a.data)
    b = float(form.b.data)
    c = float(form.c.data)
    alpha_crystal = float(form.alpha_crystal.data)
    beta_crystal = float(form.beta_crystal.data)
    gamma_crystal = float(form.gamma_crystal.data)
    is_hexagonal = form.is_hexagonal.data
    use_four_index = form.use_four_index.data
    
    #Found pole parameters
    alpha_pole = float(form.alpha_pole.data)
    beta_pole = float(form.beta_pole.data)
    cryst_rot = float(form.cryst_rot.data)
    found_pole_u = float(form.found_pole_u.data)
    found_pole_v = float(form.found_pole_v.data)
    found_pole_w = float(form.found_pole_w.data)
    found_pole = [found_pole_u, found_pole_v, found_pole_w]
    
    #Sample loading parameters
    sample_rot = float(form.sample_rot.data)
    horizontal_flip = form.horizontal_flip.data
    vertical_flip = form.vertical_flip.data
    
    #Tip/Tilt limits of the stage.
    alpha_limit = float(form.alpha_limit.data)
    beta_limit = float(form.beta_limit.data)
    superellipse_param = float(form.superellipse_param.data)
    
    #Plotting parameters
    min_beta = float(form.min_beta.data)
    max_beta = float(form.max_beta.data)
    min_alpha = float(form.min_alpha.data)
    max_alpha = float(form.max_alpha.data)
    
    planes_tip_tilt, poles_tip_tilt, stage_limits = nc.Construct_TipTiltDiagram(a, b, c, alpha_crystal, beta_crystal, gamma_crystal, \
                                                                           alpha_pole, beta_pole, cryst_rot, \
                                                                           sample_rot, horizontal_flip, vertical_flip, \
                                                                           found_pole, is_hexagonal, use_four_index, \
                                                                           alpha_limit, beta_limit, superellipse_param)
    
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
    plt_ax.set_ylim(min_beta, max_beta)
    plt_ax.set_xlim(min_alpha, max_alpha)
    plt_ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1)) #Move the legend off the diagram itself.
    plt_ax.set_xlabel('alpha')
    plt_ax.set_ylabel('beta')
    plt_ax.grid(visible=True)
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(plt_fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
    return render_template('index.html', image = pngImageB64String, form = form)
