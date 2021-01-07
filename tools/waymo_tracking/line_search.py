import os
import numpy as np 

scores = {
    0: np.arange(0.4, 0.8, 0.02),
    1: np.arange(0.4, 0.8, 0.02),
    2: np.arange(0.4, 0.8, 0.02)
}

dists = {
    0: np.arange(0.4, 0.8, 0.04),
    1: np.arange(0.1, 0.5, 0.04),
    2: np.arange(0.3, 0.7, 0.04)
}

for label in range(3):
    score_list = scores[label]
    dist_list = dists[label]

    for score in score_list:
        for dist in dist_list:
            work_dir = "waymo_track/label_{}_score_{}_max_age_{}_dist_{}".format(label, score, dist)

            cmd=("python tools/waymo_tracking/test.py " + 
                "--checkpoint /home/tianweiy/base/work_dirs/waymo_centerpoint_voxelnet_two_sweeps_3x_with_velo/prediction.pkl"
                "--work_dir {}".format(work_dir) + 
                "  --info_path data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl" + 
                "--vehicle {}  --pedestrian {}  --cyclist".format(dist, dist, dist) +
                "--score_thresh {}".format(score) + 
                "--name {}".format(label) + 
                "> {}/stats.txt ".format(work_dir)
            )[0]

            print(cmd)
            os.system(cmd)
