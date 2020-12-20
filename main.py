from src.advancedLaneFinding import AdvancedLaneFinding
import sys
import json
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def main():
    if len(sys.argv) < 2:
        print("Advanced Lane Finding\n", "Usage: python3 main.py config.json")
    else:
        config_file = sys.argv[1]
        with open(config_file) as json_file:
            configs = json.load(json_file)

            # Paths for calibration data and test data
            calibrationPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), configs["calibrationPath"])

            # Chessboard size
            chessboard = (int(configs["chessboard"]["columns"]), int(configs["chessboard"]["rows"]))

            # Thresholding parameters
            threshold = configs["threshold"]
            colorThreshold = configs["colorThreshold"]

            # Perspective Transform parameters
            perspectiveTransform = configs["perspectiveTransform"]

            # Lane Finding Parameters
            laneFinding = configs["laneFinding"]

            # Video or Images
            videoMode = eval(configs["videoMode"])
            saveMode = eval(configs["saveMode"])
            if videoMode:
                print("Processing Video")
                input_video = configs["video"]
                data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_video.split(".")[0])
                # Init Pipeline
                alf = AdvancedLaneFinding(calibrationPath, data_dir, chessboard, threshold, colorThreshold,
                                          perspectiveTransform, laneFinding, videoMode, saveMode)
                clip1 = VideoFileClip(input_video)
                project_clip = clip1.fl_image(alf.process)
                project_clip.write_videofile(input_video.split(".")[0] + "_output." + input_video.split(".")[1], audio=False)
            else:
                test_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), configs["datapath"])

                # Init Pipeline
                alf = AdvancedLaneFinding(calibrationPath, test_images_dir, chessboard, threshold, colorThreshold,
                                          perspectiveTransform, laneFinding, videoMode, saveMode)
                images = alf.getImageList()
                # setup toolbar
                print("Processing Images")
                toolbar_width = len(images)
                sys.stdout.write("[%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
                for item in images.items():
                    image_name = item[0]
                    img = item[1]
                    alf.setFileName(image_name)
                    alf.process(img)
                    # update the bar
                    sys.stdout.write("-")
                    sys.stdout.flush()
                sys.stdout.write("]\n")  # this ends the progress bar
                print("Images done.")


if __name__ == "__main__":
    main()
