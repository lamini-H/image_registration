from flask import Flask, render_template, request,jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration)
from dipy.align.transforms import(TranslationTransform3D, RigidTransform3D,AffineTransform3D)
from dipy.align.metrics import CCMetric
from dipy.viz import regtools
import matplotlib.pyplot as plt



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def perform_registration(static_image, moving_image):
    moving_img = nib.load(static_image)
    static_img = nib.load(moving_image)

    moving_data = moving_img.get_fdata()
    moving_affine = moving_img.affine
    static_data = static_img.get_fdata()
    static_affine = static_img.affine

    # #statring new registration using affine methods
    # identify = np.eye(4)
    # affine_map=AffineMap(identify,static_data.shape, static_affine,moving_data,moving_affine)
    # resampled = affine_map.transform(moving_data)
    # regtools.overlay_slices(static_data,resampled,None,2,"Template","Moving")

    # #getting the mutualinformation
    # nbins = 32
    # sampling_prop = None
    # metric = MutualInformationMetric(nbins, sampling_prop)



    metric = CCMetric(3)
    level_iters = [200, 100, 50, 25]  # Number of iterations at each level
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=50)

    mapping = sdr.optimize(static_data, moving_data,static_affine,moving_affine)
    warped_moving = mapping.transform(moving_data, 'linear')
 

    static_folder = os.path.join(app.root_path, 'static')
    image_path = os.path.join(static_folder, 'Transformed_result.png')
    image_path2 = os.path.join(static_folder,'Static_and_Moving.png')

    
    regtools.overlay_slices(static_data, warped_moving, None, 2, 'Static', 'Transformed', image_path)
    regtools.overlay_slices(static_data, moving_data, None, 2, 'Static', 'Moving',   image_path2)
    # regtools.overlay_slices(static_data, warped_moving, None, 2, 'Static', 'Transformed', 'Transformed_result.png')
    
   
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(static_data[:, :, static_data.shape[2] // 2], cmap='gray')
    # axes[0].set_title('Static Image')
    # axes[1].imshow(warped_moving[:, :, warped_moving.shape[2] // 2], cmap='gray')
    # axes[1].set_title('Transformed Moving Image')
    # plt.savefig('static/side_by_side.png')


    return warped_moving

def dice_coefficient(inputs, targets, smooth=1e-8):
     # Flatten label and prediction tensors
    inputs =inputs.astype(np.bool_)
    targets = targets.astype(np.bool_)
    
    # Calculate intersection and Dice coefficient
    intersection = np.logical_and(inputs,targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return dice


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        static_file = request.files['static_image']
        moving_file = request.files['moving_image']
        if static_file and moving_file:
            static_filename = secure_filename(static_file.filename)
            moving_filename = secure_filename(moving_file.filename)
            static_path = os.path.join(app.config['UPLOAD_FOLDER'], static_filename)
            moving_path = os.path.join(app.config['UPLOAD_FOLDER'], moving_filename)
            static_file.save(static_path)
            moving_file.save(moving_path)
            warped_moving = perform_registration(static_path, moving_path)
            # Calculate Dice coefficient
            # Load images
            static_img = nib.load(static_path)
            static_data = static_img.get_fdata()
            dice = dice_coefficient(static_data, warped_moving)
            # Render result template
            return render_template('result.html', dice=dice)
        else:
            return 'Error: Please upload both static and moving images.'

if __name__ == '__main__':
    app.run(debug=True)
