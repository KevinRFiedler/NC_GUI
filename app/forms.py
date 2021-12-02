from flask_wtf import FlaskForm
from wtforms import DecimalField, BooleanField, SubmitField

class ParameterForm(FlaskForm):
    #Parameters of the crystal system for the sample (cubic, trigonal, etc.)
    a = DecimalField(label = 'Crystal a', default = 1, places = 1)
    b = DecimalField(label = 'Crystal b', default = 1, places = 1)
    c = DecimalField(label = 'Crystal c', default = 1, places = 1)
    alpha_crystal = DecimalField(label = 'Crystal alpha', default = 90, places = 1)
    beta_crystal = DecimalField(label = 'Crystal beta', default = 90, places = 1)
    gamma_crystal = DecimalField(label = 'Crystal gamma', default = 90, places = 1)
    is_hexagonal = BooleanField(label = 'Hexagonal?', default = False) #Changes the poles/planes grouping to hexagonal vs. cubic (True/False)
    use_four_index = BooleanField(label = 'Use four index notation?', default = False) #Whether to use 3-index or 4-index notation for poles (True/False)

    #Found pole parameters
    alpha_pole = DecimalField(label = 'Found Pole alpha', default = 0, places = 1)
    beta_pole = DecimalField(label = 'Found Pole beta', default = 0, places = 1)
    cryst_rot = DecimalField(label = 'Crysal Rotation', default = 0, places = 1)
    found_pole_u = DecimalField(label = 'Found Pole u', default = 1, places = 1)
    found_pole_v = DecimalField(label = 'Found Pole v', default = 1, places = 1)
    found_pole_w = DecimalField(label = 'Found Pole w', default = 1, places = 1)

    #Sample loading parameters
    sample_rot = DecimalField(label = 'Sample rotation', default = 0, places = 1)
    horizontal_flip = BooleanField(label = 'Horizontal flip when loading?', default = False)
    vertical_flip = BooleanField(label = 'Vertical flip when loading?', default = False)

    #Tip/Tilt limits of the stage.
    alpha_limit = DecimalField(label = 'alpha limit', default = 36, places = 1)
    beta_limit = DecimalField(label = 'beta limit', default = 32, places = 1)
    superellipse_param = DecimalField(label = 'Superellipse Exponent', default = 3.3, places = 1) #adjusts the shape of the limit curve
    
    #Plotting parameters
    min_beta = DecimalField(label = 'Min beta', default = -40, places = 1)
    max_beta = DecimalField(label = 'Max beta', default = 40, places = 1)
    min_alpha = DecimalField(label = 'Min alpha', default = -40, places = 1)
    max_alpha = DecimalField(label = 'Max alpha', default = 40, places = 1)
    
    #Button we use to update the picture.
    update = SubmitField('Update')
