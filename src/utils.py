import os
import re
import numpy as np
import matplotlib.pyplot as plt


def check_dir(filename):
    # check dir: checks the path of a given filename/directory, if it doesn't exist, then create the path
    #
    # filename given filename/directory to be checked
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def my_float(s):
    constants = {"pi": "np.pi", "e": "np.e"}
    substr = " ".join(re.findall("[a-zA-Z]+", s))
    if substr in constants:
        x = s.replace(substr, str(constants[substr]))
        return eval(x)
    else:
        return float(s)


def save_image(img, file_name, path, path_extension):
    results_path = path.replace(path.split("/")[-1], "results/" + path.split("/")[-1])
    # Setup
    base_name = file_name.split("/")[-1].split(".")[0]
    extension = file_name.split("/")[-1].split(".")[1]
    output_directory = os.path.join(results_path + "_" + path_extension)
    output_image_name = output_directory + "/" + base_name + "_" + path_extension + "." + extension
    check_dir(output_image_name)
    plt.imsave(output_image_name, img, cmap='gray')
