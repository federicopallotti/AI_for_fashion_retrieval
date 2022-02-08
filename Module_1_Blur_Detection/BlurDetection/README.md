**Code for computing the blur detection on frames sampled from the input video**

To run the code edit the `config.json` file, setting the correct hyperparameters and run 'python3 main.py'.

`config.json` file :

{
  "path" : "input_video.mp4", # the input video path 
  "mode" : 2, # the modality to extract frames
  "kernel_n" : 2, # the kernel to compute the Laplacian edge detection
  "threshold" : 300, # the thresholding value of the variance, this discriminates frames.
  "parameters_mode1" : {
    "start_pts": 0, # if mode 1, the second to start
    "end_pts": 3, # if mode 1, the second to end
    "skip": 10, # if mode 1, the frames to skip
  },
  "parameters_mode2" : {
    "number_of_images" : 4,  # if mode 2, the total number of image to skip
    "seconds_to_skip" : 1,  # if mode 2, the seconds to skip between frames
    "sec_start_point" : 0  # if mode 2, the second to start
  },
  "format" : "png" # format of the saved frames.
}

kernel 1 is : [0, -1, 0]
              [-1, 4, -1]
              [0, -1, 0]

kernel 2 is : [-1, -1, -1]
              [-1, 8, -1]
              [-1, -1, -1

if mode 1 is set, mode 2 parameters are ignored and viceversa.

The `utils.py` file computes the mode 2. This mode differs from mode 1 because uses an algorithm based on the open-cv VideoCapture class.
The mode 1 uses the torchvision.io.read_video method to extract frames. This approach could be computationally heavier for a huge amount of frames.

The output folder in which the frames will be stored is `Images_inputParser`.


