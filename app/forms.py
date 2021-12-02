from flask_wtf import FlaskForm
from wtforms import DecimalField, PasswordField, SubmitField
from wtforms.validators import DataRequired

class ParameterForm(FlaskForm):
    alpha = DecimalField(label = 'Found Pole alpha', default = 0, places = 1)
    beta = DecimalField(label = 'Found Pole beta', default = 0, places = 1)
    update = SubmitField('Update')
