# Automatic Light Echo Detection (ALED)

## Installation:
1. Install Python 3
2. (Optional) create a virtual environment: `virtualenv aledpy`, and activate virtual environment: `source aledpy/bin/activate`
3. Install jupyter notebook: `pip install jupyterlab`
4. Install dependencies via `pip install ...`
### Dependencies:
`astropy==3.0.5`\
`matplotlib==3.0.2`\
`numpy==1.16.3`\
`opencv-python==3.4.4.19`\
`pandas==0.24.1`\
`scikit-image==0.14.2`\
`scikit-learn==0.20.2`\
`scipy==1.2.1`\
`tensorflow-gpu==1.11.0` or `tensorflow==1.11.0`

Installing tensorflow-gpu==1.11.0 isn't a straight forward `pip install tensorflow-gpu==1.11.0`, instead follow this tutorial https://www.tensorflow.org/install/gpu. `tensorflow==1.11.0` can be installed easily using `pip`, however, it is typically much slower than tensorflow-gpu because it only uses the cpu.

5. Run jupyter notebook: `jupyter notebook`

If you're remotely connected to the computer than port forward jupyter notebook onto your local computer via `local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host` as shown here https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh.

6. To make sure everything is installed correctly, run `test.ipynb`


## Description
The `test.ipynb` file contains sample code to get you started. Call function `classify_fits(snle_image_paths, snle_names, start)` from file `model_functions.py` to start the classification process. `snle_image_paths` is a Python list of the file paths of each differenced image to be classified (each image in .fits format). `snle_names` is a Python list of the names of the images corresponding to the file paths (names can be arbitrary strings). `start` is an int that allows you to start the classification process where you left off, in case the process has to be terminated.

The input image will be cropped to multiple 200x200 sub-images (note that padding will be added to the input image so that it is completely divisible by 200x200). Each sub-image is passed through the network for classification, and a corresponding routing path visualization image is produced. The routing path visualization images are stiched together and saved as a .png in directory `asto_package_pics/`, along with the input image.

For each .fits image, a corresponding routing path visualization image will be saved to `astro_package_pics/`. In addition, a text file titled `snle_candidates.txt` will be created. The text file contains the name of each .fits file, and 5 values called `Count1`, `Count2`, `Count3`, `Avg1`, `Avg2`, representing the liklihood of the image containing a light echo. From experience, if `Count1` is non-zero than the image should be considered a light echo candidate.

* `Count1`: A count of the number of pixels in the routing path visualization image that have a value greater than 0.00042.
* `Count2`: A count of the number of pixels in the routing path visualization image that have a value greater than 0.00037.
* `Count3`: A count of the number of pixels in the routing path visualization image that have a value greater than 0.00030.
* `Avg1`: Is the average length of the light-echo-detecting capsule for the `top_n` sub-images with the largest length for the light-echo-detecting capsule.
* `Avg2`: Is the average length of the light-echo-detecting capsule for the `small_n` sub-images with the largest length for the light-echo-detecting capsule.

As default, `top_n=45` and `small_n=10`. `top_n` and `small_n` are arguments for `classify_fits()` and can be changed via `classify_fits(..., top_n=45, small_n=10)`.


## Test Package

To check if the dependencies have been installed correctly open `test.ipynb` and run all cells. If successful, 3 files should be produced:
* `snle_candidates.txt`: Will contain the following line `test.fits 375.000000 440.000000 562.000000 0.731215 0.880019`.
* `astro_package_pics/rpv_test.fits.png`: The routing path visualization image of `test.fits`.
* `astro_package_pics/snle_test.fits.png`: `test.fits` in .png format
