import time
import numpy as np
import scipy.sparse as sp
import psutil
import time
try:
    import gurobipy as gp
except ImportError:
    pass

class SWTracker():
    """
    Sliding window tracker class.
    
    Attributes
    ----------
    sw_dets : dict
        Sliding window detections attributes with values as list(np.array) with 
        list length is T frames and array length D detections.
    x_det_ind : np.array
        Array (NxT) of detection indices for N decision variables and T frames.
        Detection index is one-based (0 for skipped detection).

    """
    def __init__(self, work_dir=None):
        self.cost_flg = {"dist_max": False, "dist_avg": False, "dist_sum": False,
                         "llr": True, "learned": False}
        assert sum(self.cost_flg.values()) == 1
        self.solver_flg = {"gurobi": True, "greedy": False}
        assert sum(self.solver_flg.values()) == 1
        self.sw_len_max = 4
        assert self.sw_len_max >= 2
        self.plot_flg = False
        self.debug_flg = True
        self.profile_flg = True
        self.work_dir = work_dir
        self.use_inactive_tracks = False
        self.frame_counter = 0
        self.prof_times = []
        self.distance_scaling_cutoff_time = 1 # seconds, TODO: change for dataset?
        self.dt = 0.5 # seconds, TODO: change for dataset

        # 99.9 percentile of l2 velocity error distribution TODO: tuning
        self.class_vel_limit = {
            'bicycle':6,
            'bus':11,
            'car':8,
            'motorcycle':26,
            'pedestrian':2,
            'trailer':6,
            'truck':8,
            }
        self.reset()


    def reset(self):
        """ Reset tracker state."""
        self.sw_dets = {"delta_t": [], "pos_x": [], "pos_y": [], "vel_x": [],
                        "vel_y": [], "class_id": [], "score": [], "max_vel": [],
                        "size_w": [], "size_l": [], "rot": [],
                        "num_dets": np.array([], dtype=np.int16), "indices": [],
                        }
        self.scene_frame_counter = 0
        self.sw_len_curr = 0
        self.x_det_ind = None


    def expand_window(self, detections, time_lag):
        ''' 
        Append detections to sliding window history.
        Args:
            detections (list[dict]): List of detection dictionaries.
                sample_token (str): Sample token.
                translation (np.array): Translation in meters with shape (3).
                size (np.array): Size in meters with shape (3).
                rotation (np.array): Rotation quaternion with shape (4).
                velocity (np.array): Velocity in meters with shape (3).
                detection_name (str): Predicted class name.
                detection_score (float): Predicted class score.
                attribute_name (str): Predicted attribute name.
                ct (np.array): Center position in meters with shape (2).
                tracking (np.array): Translation to past frame with shape (2).
                label_preds (int): Predicted class label.
            time_lag (float): Timestep from last assignment to current one.
        '''
        profiler = self.profile_start()
        self.sw_dets['delta_t'].append(time_lag)
        self.sw_dets['pos_x'].append(
            np.array([det['translation'][0] for det in detections], np.float32))
        self.sw_dets['pos_y'].append(
            np.array([det['translation'][1] for det in detections], np.float32))
        if self.sw_dets['vel_x'] == []:
            self.sw_dets['vel_x'].append(
                np.zeros(len(detections), np.float32))
            self.sw_dets['vel_y'].append(
                np.zeros(len(detections), np.float32))
        else:
            self.sw_dets['vel_x'].append(
                np.array([det['velocity'][0] for det in detections], np.float32))
            self.sw_dets['vel_y'].append(
                np.array([det['velocity'][1] for det in detections], np.float32))
        self.sw_dets['class_id'].append(
            np.array([det['label_preds'] for det in detections], np.int16))
        self.sw_dets['score'].append(
            np.array([det['detection_score'] for det in detections], np.float32))
        self.sw_dets['size_w'].append(
            np.array([det['size'][0] for det in detections], np.float32))
        self.sw_dets['size_l'].append(
            np.array([det['size'][1] for det in detections], np.float32))
        self.sw_dets['rot'].append(
            np.array([det['rotation'][2] for det in detections], np.float32))
        self.sw_dets['max_vel'].append(
            np.array([self.class_vel_limit[det['detection_name']]
                for det in detections], np.float32))
        self.sw_dets['indices'].append(np.arange(len(detections),
                                                 dtype=np.int16))
        self.sw_dets['num_dets'] = np.append(
            self.sw_dets['num_dets'], len(detections))
        self.sw_len_curr += 1
        self.frame_counter += 1
        self.scene_frame_counter += 1
        self.profile_end(profiler, 'expand_window')


    def contract_window(self, tracks):
        ''' 
        Remove and insert inactive tracks at beginning of sliding window.
        Args:
            tracks (list[dict]): List of track dictionaries.
                sample_token (str): Sample token.
                translation (np.array): Translation in meters with shape (3).
                size (np.array): Size in meters with shape (3).
                rotation (np.array): Rotation quaternion with shape (4).
                velocity (np.array): Velocity in meters with shape (3).
                detection_name (str): Predicted class name.
                detection_score (float): Predicted class score.
                attribute_name (str): Predicted attribute name.
                ct (np.array): Center position in meters with shape (2).
                tracking (np.array): Tracking state with shape (2).
                label_preds (int): Predicted class label.
                tracking_id (int): Tracking ID.
                age (int): Age of track.
                active (int): Track is active or not.
                detection_ids (list[int]): List of track history detection IDs.
        '''
        profiler = self.profile_start()
        # Contract window # TODO: most efficient way to pop list and aray?
        if self.sw_len_curr > self.sw_len_max:
            self.sw_len_curr -= 1
            self.sw_dets['delta_t'].pop(0)
            self.sw_dets['pos_x'].pop(0)
            self.sw_dets['pos_y'].pop(0)
            self.sw_dets['vel_x'].pop(0)
            self.sw_dets['vel_y'].pop(0)
            self.sw_dets['class_id'].pop(0)
            self.sw_dets['score'].pop(0)
            self.sw_dets['max_vel'].pop(0)
            self.sw_dets['indices'].pop(0)
            self.sw_dets['num_dets'] = self.sw_dets['num_dets'][1:]
            self.x_det_ind = np.delete(self.x_det_ind, 0, axis=1)

        # Replace detections with tracks at beginning of window
        # TODO: remove
        if self.use_inactive_tracks:
            pos_x = []
            pos_y = []
            vel_x = []
            vel_y = []
            class_id = []
            score = []
            max_vel = []
            for track in tracks:
                if track['age'] >= self.sw_len_curr - 1:
                    time_since_window = track['t_age'] - np.sum(
                        self.sw_dets['delta_t'][1:-1])
                    pos_x.append(track['translation'][0]
                                 - track['velocity'][0]*time_since_window)
                    pos_y.append(track['translation'][1]
                                 - track['velocity'][1]*time_since_window)
                    vel_x.append(track['velocity'][0])
                    vel_y.append(track['velocity'][1])
                    class_id.append(track['label_preds'])
                    score.append(track['detection_score'])
                    max_vel.append(self.class_vel_limit[track['detection_name']])
            self.sw_dets['pos_x'][0] = np.array(pos_x, np.float32)
            self.sw_dets['pos_y'][0] = np.array(pos_y, np.float32)
            self.sw_dets['vel_x'][0] = np.array(vel_x, np.float32)
            self.sw_dets['vel_y'][0] = np.array(vel_y, np.float32)
            self.sw_dets['class_id'][0] = np.array(class_id, np.int32)
            self.sw_dets['score'][0] = np.array(score, np.float32)
            self.sw_dets['max_vel'][0] = np.array(max_vel, np.float32)
            self.sw_dets['indices'][0] = np.arange(len(tracks) + 1)
            self.sw_dets['num_dets'][0] = len(tracks)
            self.profile_end(profiler, 'contract_window')


    def get_detection_to_track_map(self):
        '''
        Get mapping of detection indices to track decision variables.
        '''
        profiler = self.profile_start()
        sw_num_dets = self.sw_dets['num_dets'][-self.sw_len_curr:]
        sw_indices = [np.arange(self.sw_dets['num_dets'][i]+1, dtype=np.int16)
                      for i in range(-self.sw_len_curr, 0)]
        # Change detection indices to one-based (0 for skipped detection)
        
        # Old track hypotheses and new detections
        if self.x_det_ind is not None:
            x_old = np.unique(self.x_det_ind, axis=0)
            x_old_len = x_old.shape[0]
            ip_len_x = int(x_old_len*(sw_num_dets[-1] + 1) +
                           (sw_num_dets[-1])*np.sum(sw_num_dets[:-1]))
            x_det_ind = np.zeros((ip_len_x, self.sw_len_curr), dtype=np.int16)
            x_det_ind[:x_old_len*(sw_num_dets[-1] + 1), :-1] = \
                np.tile(x_old, (sw_num_dets[-1]+1, 1))
            x_det_ind[:x_old_len*(sw_num_dets[-1] + 1), -1] = \
                np.repeat(sw_indices[-1], x_old_len)

        # New track hypotheses from pairs of old and new detections
        if self.x_det_ind is None:
            ip_len_x = int(np.prod(sw_num_dets))
            x_det_ind = np.zeros((ip_len_x, self.sw_len_curr), dtype=np.int16)
            start_ind = 0
        else:
            start_ind = x_old_len*(sw_num_dets[-1] + 1)
        iter_start_ind = start_ind
        for iter_frame in range(-self.sw_len_curr, -1):
            iter_end_ind = iter_start_ind + (sw_num_dets[-1])*(
                sw_num_dets[iter_frame])
            x_det_ind[iter_start_ind:iter_end_ind, iter_frame] = \
                np.repeat(sw_indices[iter_frame][1:],
                            sw_num_dets[-1])
            iter_start_ind = iter_end_ind
        assert iter_end_ind == ip_len_x
        x_det_ind[start_ind:, -1] = \
            np.tile(sw_indices[-1][1:], np.sum(sw_num_dets[:-1]))

        # Create tracks with all permuations of detection indices
        # sw_num_dets = self.sw_dets['num_dets'][-self.sw_len_curr:]
        # ip_len_x = int(np.prod(sw_num_dets + 1))
        # x_det_ind = np.zeros((ip_len_x, self.sw_len_curr), dtype=np.int16)
        # for iter_frame in range(-self.sw_len_curr, -1):
        #     x_det_ind[:, iter_frame] = np.tile(np.repeat(
        #         self.sw_dets['indices'][iter_frame],
        #         np.prod(sw_num_dets[iter_frame+1:] + 1)),
        #         np.prod(sw_num_dets[:iter_frame] + 1))
        # x_det_ind[:, -1] = np.tile(self.sw_dets['indices'][-1],
        #     np.prod(sw_num_dets[:-1] + 1))
        self.profile_end(profiler, 'get_detection_to_track_map')

        return x_det_ind


    def filter_map(self, x_det_ind):
        """Filter tracks based on class, score, and distance."""
        def filter_det_count(x_det_ind):
            """Filter out tracks with less than 2 detections"""
            # Detection index = 0 means no detection (frame skipped)
            profiler = self.profile_start()
            sw_num_dets = self.sw_dets['num_dets'][-self.sw_len_curr:]
            t_bool = x_det_ind>0
            t_sum = np.zeros(np.shape(t_bool[:, 0]), dtype=np.int16)
            for i in range(np.shape(t_bool)[1]):
                t_sum = t_sum + t_bool[:, i]
            t_bool = t_sum > 1
            x_det_ind = x_det_ind[np.where(t_bool)]
            self.profile_end(profiler, 'filter_det_count')

            return x_det_ind

        def filter_class(sw_len_curr, sw_dets, x_det_ind):
            """Filter tracks with more than one class"""
            profiler = self.profile_start()
            x_class_ids = np.ones(x_det_ind.shape, dtype=np.int16)*-1
            for col_id in range(-sw_len_curr, 0):
                row_ids = x_det_ind[:, col_id] > 0
                detection_indices = x_det_ind[row_ids, col_id] -1
                x_class_ids[row_ids, col_id] = \
                    sw_dets['class_id'][col_id][detection_indices]
            x_class_sorted = np.sort(x_class_ids, axis=1)
            x_class_count = (x_class_sorted[:, 1:] != x_class_sorted[:, :-1]
                            ).sum(axis=1) + 1
            x_cond_1class_det = np.all(
                np.stack((x_class_count == 1, x_class_sorted[:, 0] != -1)), axis=0)
            x_cond_2class_nodet = np.all(
                np.stack((x_class_count == 2, x_class_sorted[:, 0] == -1)), axis=0)
            x_cond_check = np.any(
                np.stack((x_cond_1class_det, x_cond_2class_nodet)), axis=0)
            x_cond_1class_det = (x_class_count==1)*(x_class_sorted[:, 0]!=-1)
            x_cond_2class_nodet = (x_class_count==2)*(x_class_sorted[:, 0]==-1)
            x_class_cond = x_cond_1class_det+x_cond_2class_nodet
            x_det_ind = x_det_ind[x_class_cond, :]
            self.profile_end(profiler, 'filter_class')

            return x_det_ind


        def filter_distance_coarse(sw_len_curr, sw_dets, x_det_ind):
            """Filter tracks with large distance neglecting velocity"""
            profiler = self.profile_start()
            x_pos_x = np.zeros(x_det_ind.shape, dtype=np.float32)
            x_pos_y = np.zeros(x_det_ind.shape, dtype=np.float32)
            for col_id in range(-sw_len_curr, 0):
                row_ids = x_det_ind[:, col_id] > 0
                detection_indices = x_det_ind[row_ids, col_id] -1
                x_pos_x[row_ids, col_id] = \
                    sw_dets['pos_x'][col_id][detection_indices]
                x_pos_y[row_ids, col_id] = \
                    sw_dets['pos_y'][col_id][detection_indices]
            
            # Get max distance between detections averaged by number of frames
            num_row = np.size(x_det_ind, 0)
            num_col = np.size(x_det_ind, 1)
            x_dist_max = np.zeros(num_row, dtype=np.float32)
            x_diff_x = x_pos_x[:,1:]-x_pos_x[:,:-1]
            x_diff_y = x_pos_y[:,1:]-x_pos_y[:,:-1]
            x_bool = x_det_ind != 0
            x_diff_bool = x_bool[:,1:]*x_bool[:,:-1]
            for i in range(num_col-2):
                x_diff_x = np.hstack((x_diff_x,
                                      (x_pos_x[:,2+i:]-x_pos_x[:,:-2-i])/(2+i)))
                x_diff_y = np.hstack((x_diff_y,
                                      (x_pos_y[:,2+i:]-x_pos_y[:,:-2-i])/(2+i)))
                x_diff_bool = np.hstack((x_diff_bool,
                                         x_bool[:,2+i:]*x_bool[:,:-2-i]))
            x_dist = np.float32(np.sqrt(np.float32(x_diff_x)**2 + np.float32(x_diff_y)**2))
            x_dist_max = np.max(x_dist, where=x_diff_bool, initial=0, axis=1)

            # Get max velocity averaged by number of frames
            x_dist_limit = 140 / 3.6 * 0.5 # 140 km/h
            x_cond = x_dist_max < x_dist_limit
            x_det_ind = x_det_ind[x_cond, :]
            self.x_det_ind = x_det_ind
            self.profile_end(profiler, 'filter_distance_coarse')

            return x_det_ind


        def filter_distance(sw_len_curr, sw_dets, x_det_ind, max_pair_frames=1):
            """Filter tracks with large distance"""
            profiler = self.profile_start()
            x_pos_x = np.zeros(x_det_ind.shape, dtype=np.float32)
            x_pos_y = np.zeros(x_det_ind.shape, dtype=np.float32)
            x_vel_x = np.zeros(x_det_ind.shape, dtype=np.float32)
            x_vel_y = np.zeros(x_det_ind.shape, dtype=np.float32)
            x_max_vel = np.zeros(x_det_ind.shape, dtype=np.float32)
            for col_id in range(-sw_len_curr, 0):
                row_ids = x_det_ind[:, col_id] > 0
                detection_indices = x_det_ind[row_ids, col_id] -1
                x_pos_x[row_ids, col_id] = \
                    sw_dets['pos_x'][col_id][detection_indices]
                x_pos_y[row_ids, col_id] = \
                    sw_dets['pos_y'][col_id][detection_indices]
                x_vel_x[row_ids, col_id] = \
                    sw_dets['vel_x'][col_id][detection_indices]
                x_vel_y[row_ids, col_id] = \
                    sw_dets['vel_y'][col_id][detection_indices]
                x_max_vel[row_ids, col_id] = \
                    sw_dets['max_vel'][col_id][detection_indices]

            map_shape = np.shape(x_det_ind)
            num_row = np.size(x_det_ind, 0)
            num_col = np.size(x_det_ind, 1)
            x_dist_max = np.zeros(num_row, dtype=np.float32)
            # x_frames = np.zeros(num_row, dtype=np.float32)
            # loop over all possible pairs of detections across window
            for col_a in range(0, num_col-1):
                for col_b in range(col_a+1, num_col):
                    # Select rows with both detections and none in between
                    map_bool = np.zeros(map_shape, dtype=np.bool)
                    map_cond = np.zeros(map_shape, dtype=np.bool)
                    map_bool[:, col_a:col_b+1] = x_det_ind[:, col_a:col_b+1] > 0
                    map_cond[:, col_a] = True
                    map_cond[:, col_b] = True
                    row_bool = np.all(map_bool == map_cond, axis=1)

                    # Calculate time step and number of frames between detections
                    sw_delta_t = [sw_dets['delta_t'][col_id] for col_id in
                                range(-sw_len_curr, 0)]
                    delta_t = np.sum(sw_delta_t[col_a+1:col_b+1])
                    pair_frames = (col_b-col_a)*np.ones(
                        (num_row), dtype=np.uint8)[row_bool]
                    # x_frames[row_bool] = x_frames[row_bool]\
                    #     + pair_frames

                    # Compute distance between detections for given pair of frames
                    pair_dist_x = x_pos_x[row_bool, col_b] - \
                        x_vel_x[row_bool, col_b] * delta_t - \
                        x_pos_x[row_bool, col_a]
                    pair_dist_y = x_pos_y[row_bool, col_b] - \
                        x_vel_y[row_bool, col_b] * delta_t - \
                        x_pos_y[row_bool, col_a]
                    pair_dist = np.sqrt(pair_dist_x**2 + pair_dist_y**2)
                    # x_dist_max[row_bool] = np.amax(np.stack(
                    #     [x_dist_max[row_bool], pair_dist],
                    #     axis=1), axis=1)
                    pair_dist_scaled = pair_dist / np.minimum(pair_frames, max_pair_frames)
                    x_dist_max[row_bool] = np.amax(np.stack(
                        [x_dist_max[row_bool], pair_dist_scaled],
                        axis=1), axis=1)

            x_dist_limit = np.max(x_max_vel, axis=1) * 0.5 # TODO: change 0.5 time step
            x_cond = x_dist_max < x_dist_limit
            x_det_ind = x_det_ind[x_cond, :]
            self.profile_end(profiler, 'filter_distance')

            return x_det_ind

        profiler = self.profile_start()
        max_pair_frames = self.distance_scaling_cutoff_time / self.dt
        max_batch_size = 1e8
        # x_det_ind = x_det_ind[x_det_ind[:,-1] != 0] # clipping last frame
        num_batches = int(np.ceil(x_det_ind.size / max_batch_size))
        x_det_ind_split = np.array_split(x_det_ind, num_batches, axis=0)
        for i in range(num_batches):
            if num_batches > 1:
                print(f'>> Batch: {i+1}/{num_batches}')
            print(f'>> Vars: {x_det_ind_split[i].shape[0]:,}')
            x_det_ind_split[i] = filter_class(
                self.sw_len_curr, self.sw_dets, x_det_ind_split[i])
            print(f'>> Vars: {x_det_ind_split[i].shape[0]:,}')
            x_det_ind_split[i] = filter_det_count(x_det_ind_split[i])
            print(f'>> Vars: {x_det_ind_split[i].shape[0]:,}')
            x_det_ind_split[i] = filter_distance_coarse(
                self.sw_len_curr, self.sw_dets, x_det_ind_split[i])
            print(f'>> Vars: {x_det_ind_split[i].shape[0]:,}')
            x_det_ind_split[i] = filter_distance(
                self.sw_len_curr, self.sw_dets, x_det_ind_split[i], max_pair_frames)
            print(f'>> Vars: {x_det_ind_split[i].shape[0]:,}')
        x_det_ind = np.concatenate(x_det_ind_split, axis=0)
        x_det_ind = np.unique(x_det_ind, axis=0)
        print(f'>> Vars: {x_det_ind.shape[0]:,}')
        self.profile_end(profiler, 'filter_map')

        return x_det_ind


    def get_track_detection_states(self, x_det_ind):
        """
        Get states for the detections of each track decision variable.

        Args:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
                Detection index is one-based (0 for skipped detection).
        
        Returns:
            x_det_states (dict[np.array[NxT]]): dictionary of detection states, 
                each state is an array for N decision variables and T frames.
        """
        profiler = self.profile_start()
        x_det_states = {}
        x_det_states['class_id'] = np.ones(x_det_ind.shape, dtype=np.int16)*-1
        x_det_states['pos_x'] = np.zeros(x_det_ind.shape, dtype=np.float32)
        x_det_states['pos_y'] = np.zeros(x_det_ind.shape, dtype=np.float32)
        x_det_states['vel_x'] = np.zeros(x_det_ind.shape, dtype=np.float32)
        x_det_states['vel_y'] = np.zeros(x_det_ind.shape, dtype=np.float32)
        x_det_states['max_vel'] = np.zeros(x_det_ind.shape, dtype=np.float32)
        x_det_states['score'] = np.zeros(x_det_ind.shape, dtype=np.float32)

        for col_id in range(-self.sw_len_curr, 0):
            row_ids = x_det_ind[:, col_id] > 0
            detection_indices = x_det_ind[row_ids, col_id] -1
            x_det_states['class_id'][row_ids, col_id] = \
                self.sw_dets['class_id'][col_id][detection_indices]
            x_det_states['pos_x'][row_ids, col_id] = \
                self.sw_dets['pos_x'][col_id][detection_indices]
            x_det_states['pos_y'][row_ids, col_id] = \
                self.sw_dets['pos_y'][col_id][detection_indices]
            x_det_states['vel_x'][row_ids, col_id] = \
                self.sw_dets['vel_x'][col_id][detection_indices]
            x_det_states['vel_y'][row_ids, col_id] = \
                self.sw_dets['vel_y'][col_id][detection_indices]
            x_det_states['max_vel'][row_ids, col_id] = \
                self.sw_dets['max_vel'][col_id][detection_indices]
            x_det_states['score'][row_ids, col_id] = \
                self.sw_dets['score'][col_id][detection_indices]
        self.profile_end(profiler, 'get_track_detection_states')

        return x_det_states


    def get_track_states(self, x_det_ind, x_det_states):
        """
        Get states for each track decision variable.

        Args:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
            x_det_states (dict[np.array[NxT]]): dictionary of detection states,
                each state is an array for N decision variables and T frames.
    
        Returns:
            x_states (dict[np.array[N]]): dictionary of track states,
                each state is an array for N decision variables.
        """
        profiler = self.profile_start()
        x_states = {}

        # Get track class counts from class indices (class_id=-1 for no det)
        x_class_sorted = np.sort(x_det_states['class_id'], axis=1)
        x_class_count = (x_class_sorted[:, 1:] != x_class_sorted[:, :-1]
                         ).sum(axis=1) + 1
        x_cond_1class_det = np.all(
            np.stack((x_class_count == 1, x_class_sorted[:, 0] != -1)), axis=0)
        x_cond_2class_nodet = np.all(
            np.stack((x_class_count == 2, x_class_sorted[:, 0] == -1)), axis=0)
        x_states['one_class_bool'] = np.any( # True: one class exists in track
            np.stack((x_cond_1class_det, x_cond_2class_nodet)), axis=0)

        # Get track class indices
        x_class_vec = -1*np.ones(x_det_ind.shape[0], dtype=np.int16)
        x_class_vec[x_cond_1class_det] = x_class_sorted[x_cond_1class_det, 0]
        x_class_vec[x_cond_2class_nodet] = x_class_sorted[x_cond_2class_nodet, 1]
        x_states['class_id'] = x_class_vec # -1: multi-class, 0: class 0, 1: class 1

        # Get distance limit vector
        class_vel_limits = list(self.class_vel_limit.values())
        class_vel_limits.append(0)
        class_dist_limits = np.array(class_vel_limits)*0.5 # TODO: change 0.5 time step
        x_dist_limit_vec = class_dist_limits[x_class_vec]
        x_states['dist_limit'] = x_dist_limit_vec

        map_shape = np.shape(x_det_ind)
        num_row = np.size(x_det_ind, 0)
        num_col = np.size(x_det_ind, 1)
        x_states['dist_sum'] = np.zeros(num_row, dtype=np.float32)
        x_states['dist_sqr_sum'] = np.zeros(num_row, dtype=np.float32)
        x_states['neg_dist_sum_scaled'] = np.zeros(num_row, dtype=np.float32)
        x_states['dist_max'] = np.zeros(num_row, dtype=np.float32)
        x_states['dist_max_scaled'] = np.zeros(num_row, dtype=np.float32)
        x_states['frames'] = np.zeros(num_row, dtype=np.int16)
        # loop over all possible pairs of detections across window
        for col_a in range(0, num_col-1):
            for col_b in range(col_a+1, num_col):
                # Select rows with both detections and none in between
                map_bool = np.zeros(map_shape, dtype=np.bool)
                map_cond = np.zeros(map_shape, dtype=np.bool)
                map_bool[:, col_a:col_b+1] = x_det_ind[:, col_a:col_b+1] > 0
                map_cond[:, col_a] = True
                map_cond[:, col_b] = True
                row_bool = np.all(map_bool == map_cond, axis=1)

                # Calculate time step and number of frames between detections
                sw_delta_t = [self.sw_dets['delta_t'][col_id] for col_id in
                               range(-self.sw_len_curr, 0)]
                delta_t = np.sum(sw_delta_t[col_a+1:col_b+1])
                pair_frames = (col_b-col_a)*np.ones(
                    (num_row), dtype=np.uint8)[row_bool]
                x_states['frames'][row_bool] = x_states['frames'][row_bool]\
                    + pair_frames

                # Compute distance between detections for given pair of frames
                # pair_dist_x = x_det_states['pos_x'][row_bool, col_b] - \
                #     (x_det_states['vel_x'][row_bool, col_a] +
                #      x_det_states['vel_x'][row_bool, col_b]) / 2 * delta_t - \
                #     x_det_states['pos_x'][row_bool, col_a]
                # pair_dist_y = x_det_states['pos_y'][row_bool, col_b] - \
                #     (x_det_states['vel_y'][row_bool, col_a] +
                #      x_det_states['vel_y'][row_bool, col_b]) / 2 * delta_t - \
                #     x_det_states['pos_y'][row_bool, col_a]
                if self.scene_frame_counter - num_col + col_a == 0:
                    pair_delta_x = x_det_states['vel_x'][row_bool, col_b] * delta_t
                    pair_delta_y = x_det_states['vel_y'][row_bool, col_b] * delta_t
                else:
                    pair_delta_x = (x_det_states['vel_x'][row_bool, col_b] +
                                    x_det_states['vel_x'][row_bool, col_a]) / 2 * delta_t
                    pair_delta_y = (x_det_states['vel_y'][row_bool, col_b] +
                                    x_det_states['vel_y'][row_bool, col_a]) / 2 * delta_t
                pair_dist_x = x_det_states['pos_x'][row_bool, col_b] - \
                    pair_delta_x - x_det_states['pos_x'][row_bool, col_a]
                pair_dist_y = x_det_states['pos_y'][row_bool, col_b] - \
                    pair_delta_y - x_det_states['pos_y'][row_bool, col_a]
                pair_dist = np.sqrt(pair_dist_x**2 + pair_dist_y**2)
                scale_max = 15 # TODO: dataset specific
                pair_neg_dist_scaled = (scale_max - pair_dist)/scale_max

                # Compute running total distance, max distance, scores
                x_states['dist_sum'][row_bool] = x_states['dist_sum'][row_bool]\
                    + pair_dist
                x_states['dist_sqr_sum'][row_bool] = x_states['dist_sqr_sum'][row_bool]\
                    + pair_dist**2
                x_states['dist_max'][row_bool] = np.amax(np.stack(
                    [x_states['dist_max'][row_bool], pair_dist/pair_frames],
                    axis=1), axis=1)
                # x_states['neg_dist_sum_scaled'][row_bool] = \
                #     x_states['neg_dist_sum_scaled'][row_bool] + pair_neg_dist_scaled
                x_states['dist_max_scaled'][row_bool] = np.amax(np.stack(
                    [x_states['dist_max_scaled'][row_bool],
                     pair_neg_dist_scaled/pair_frames], axis=1), axis=1)

        x_states['dist_avg'] = x_states['dist_sum']/x_states['frames']
        x_states['neg_dist_avg_scaled'] = (scale_max-x_states['dist_sum'])/(
            scale_max*x_states['frames'])
        x_states['score_sum'] = np.sum(x_det_states['score'], axis=1)
        x_states['score_log'] = np.log(x_det_states['score'],
                                       out=np.zeros_like(x_det_states['score']),
                                       where=x_det_states['score']!=0)
        x_states['score_sum_log'] = np.sum(x_states['score_log'], axis=1)
        x_states['score_avg'] = x_states['score_sum']/(x_states['frames']+1)
        x_states['score_min'] = np.min(x_det_states['score'],
            where=x_det_states['score']!=0, initial = 1, axis=1)
        x_states['skip_count'] = np.sum(x_det_ind == 0, axis=1)
        x_states['det_count'] = np.sum(x_det_ind != 0, axis=1)
        self.profile_end(profiler, 'get_track_states')

        return x_states


    def get_cost_vec(self, x_states):
        """
        Get cost vector for detections in format of x.

        Args:
            x_states (dict[np.array[N]]): dictionary of track states,
                each state is an array for N decision variables.
        
        Returns:
            cost_vec (np.array[Nx1]): Cost vector for N decision variables.
        """
        profiler = self.profile_start()
        # TODO: add other costs
        # Euclidean distance between det_a and det_b back projected with -vel
        if self.cost_flg['dist_max']:
            cost_vec = -x_states['dist_max']
            cost_vec = cost_vec - cost_vec.min()
            cost_vec = cost_vec/cost_vec.max() + 1e-6
        if self.cost_flg['dist_avg']:
            cost_vec = -x_states['dist_avg']
            cost_vec = cost_vec - cost_vec.min()
            cost_vec = cost_vec/cost_vec.max() + 1e-6
        if self.cost_flg['dist_sum']:
            cost_vec = x_states['neg_dist_sum_scaled']
        # Probabilistic log likelihood ratio
        if self.cost_flg['llr']:
            const_cost = 100.0*x_states['det_count']
            dist_cost = -0.5*x_states['dist_sqr_sum']
            signal_cost = 5.0*x_states['score_sum_log']
            skip_cost = -90.0*x_states['skip_count']
            cost_vec = const_cost + dist_cost + signal_cost + skip_cost
        # Learned affinity
        elif self.cost_flg['learned']:
            raise NotImplementedError('Learned affinity not implemented yet.')
        x_states['cost_vec'] = cost_vec
        self.profile_end(profiler, 'get_cost_vec')
        return cost_vec


    def filter_hypotheses(self, x_det_ind, x_det_states, x_states, cost_vec):
        """
        Filter out decision variables that are improbable.
        
        Args:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
            x_det_states (dict[np.array[NxT]]): dictionary of detection states,
                each state is an array for N decision variables and T frames.
            x_states (dict[np.array[N]]): dictionary of track states,
                each state is an array for N decision variables.
            
        Returns:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
            x_det_states (dict[np.array[NxT]]): dictionary of detection states,
                each state is an array for N decision variables and T frames.
            x_states (dict[np.array[N]]): dictionary of track states,
                each state is an array for N decision variables.
        """
        profiler = self.profile_start()
        # Limit to only top hypotheses per detection
        max_hyp = 200
        max_det_ind = np.max(x_det_ind[:,-1])
        x_ind = []
        hyp_count = []
        for i in range(0, max_det_ind+1):
            det_ind = np.flatnonzero(x_det_ind[:, -1] == i)
            hyp_count.append(len(det_ind))
            if len(det_ind) > max_hyp:
                top_det_ind = np.argpartition(cost_vec[det_ind], -max_hyp)[-max_hyp:]
                det_ind = det_ind[top_det_ind]
            x_ind.append(det_ind)
        x_ind = np.concatenate(x_ind)
        x_states = {key: val[x_ind] for key, val in x_states.items()}
        x_det_ind = x_det_ind[x_ind, :]
        x_det_states = {key: val[x_ind, :] for key, val in x_det_states.items()}
        cost_vec = cost_vec[x_ind]
        print(f'>> Hypotheses: {hyp_count}')
        print(f'>> Vars: {x_det_ind.shape[0]:,}')
        self.profile_end(profiler, 'filter_hypotheses')

        return x_det_ind, x_det_states, x_states, cost_vec


    def get_constraints(self, x_det_ind):
        """
        Get constraints for optimization problem in form Ax<=b.

        Args:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables.
        
        Returns:
            A (scipy.sparse.csr_matrix): Constraint matrix.
            b (np.array[N]): Constraint vector.
        """
        profiler = self.profile_start()
        # Compute number of nonzero elements in A matrix
        ip_len_x = x_det_ind.shape[0]
        sw_num_dets = self.sw_dets['num_dets'][-self.sw_len_curr:]
        # nnz_a_mat = self.sw_len_curr * ip_len_x
        # for i in range(0, self.sw_len_curr):
        #     nnz_a_mat = nnz_a_mat - int(np.prod(sw_num_dets + 1) /
        #                                 (sw_num_dets[i] + 1) -
        #                                 np.sum(sw_num_dets) +
        #                                 sw_num_dets[i] - 1)
        row_ind_a_mat = []
        col_ind_a_mat = []
        row_a_ind = 0
        # frame_ind_vec_from_ip_a_row_ind = np.zeros(np.sum(sw_num_dets))
        # det_ind_vec_from_ip_a_row_ind = np.zeros(np.sum(sw_num_dets))
        for frame_ind in range(0, self.sw_len_curr):
            for det_ind in range(1, sw_num_dets[frame_ind] + 1):
                var_ind_vec = np.flatnonzero(x_det_ind[:, frame_ind] == det_ind)
                # frame_ind_vec_from_ip_a_row_ind[row_a_ind] = frame_ind
                # det_ind_vec_from_ip_a_row_ind[row_a_ind] = det_ind
                for var_ind in range(0, var_ind_vec.size):
                    row_ind_a_mat.append(row_a_ind)
                    col_ind_a_mat.append(var_ind_vec[var_ind])
                row_a_ind = row_a_ind + 1
        det_count_in_window = np.sum(sw_num_dets)
        nnz_a_mat = len(row_ind_a_mat)
        data_a_mat = np.ones(nnz_a_mat)
        row_ind_a_mat = np.array(row_ind_a_mat)
        col_ind_a_mat = np.array(col_ind_a_mat)

        ip_a_mat = sp.csr_matrix((data_a_mat, (row_ind_a_mat, col_ind_a_mat)),
                                 shape=(det_count_in_window, ip_len_x))
        ip_b_vec = np.ones(det_count_in_window)
        test = ip_a_mat.toarray()
        self.profile_end(profiler, 'get_constraints')

        return ip_a_mat, ip_b_vec


    def solve_ip(self, ip_a_mat, ip_b_vec, ip_c_vec, x_det_states, x_det_ind, x_states, debug_flg=False):
        """
        Solve integer program.
        
        Args:
            ip_A_mat (np.array[MxN]): Matrix of constraints.
            ip_b_vec (np.array[Mx1]): Vector of constraints.
            ip_c_vec (np.array[Nx1]): Cost vector.
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
            x_det_states (dict[np.array[NxT]]): dictionary of detection states, 
                each state is an array for N decision variables and T frames.
            debug_flg (bool): Debug flag.

        Returns:
            ip_x_sol (np.array[Nx1]): Solution to integer program.
        """
        profiler = self.profile_start()
        if self.solver_flg['gurobi']:
            ip_x_sol = 0
            try:
                # Optimize model and extract solution
                start_time = time.time()
                ip_model = gp.Model("model")
                ip_len = ip_c_vec.shape[0]
                ip_x = ip_model.addMVar(shape=ip_len, vtype=gp.GRB.BINARY, name="variables")
                ip_model.setObjective(ip_c_vec @ ip_x, gp.GRB.MAXIMIZE)
                ip_model.addConstr(ip_a_mat @ ip_x <= ip_b_vec, name="constraints")
                ip_model.setParam('OutputFlag', 0)
                ip_model.optimize()
                ip_x_sol = ip_x.X.astype(int)
                if debug_flg:
                    print(f"-- Optimization: {round((time.time() - start_time), 3)}"
                        "s seconds --")
                ip_sol_status = ip_model.getAttr(gp.GRB.Attr.Status)
                if ip_sol_status == 2:
                    if debug_flg:
                        print('Status code = 2. OPTIMAL. '
                            'Model was solved to optimality (subject to tolerances)' 
                            ' and an optimal solution is available. ')
                else:
                    print('Status code =', ip_sol_status,
                          '. OFF-NOMINAL. POSSIBLE ISSUE.')
            except gp.GurobiError as exception:
                print('Error code: ' + str(exception))
            except AttributeError:
                print('Encountered an attribute error')

            # lp_x_sol = 0
            # try:
            #     # Optimize model and extract solution
            #     start_time = time.time()
            #     lp_model = gp.Model("model")
            #     ip_len = ip_c_vec.shape[0]
            #     lp_x = lp_model.addMVar(shape=ip_len, lb=0.0, ub=1.0,
            #                             vtype='C', name="variables")
            #     lp_model.setObjective(ip_c_vec @ lp_x, gp.GRB.MAXIMIZE)
            #     lp_model.addConstr(ip_a_mat @ lp_x <= ip_b_vec, name="constraints")
            #     lp_model.setParam('OutputFlag', 1)
            #     lp_model.optimize()
            #     lp_x_sol = lp_x.X.astype(int)
            #     if debug_flg:
            #         print(f"-- Optimization: {round((time.time() - start_time), 3)}"
            #             "s seconds --")
            #     lp_sol_status = lp_model.getAttr(gp.GRB.Attr.Status)
            #     if lp_sol_status == 2:
            #         if debug_flg:
            #             print('Status code = 2. OPTIMAL. '
            #                 'Model was solved to optimality (subject to tolerances)' 
            #                 ' and an optimal solution is available. ')
            #     else:
            #         print('Status code =', lp_sol_status,
            #               '. OFF-NOMINAL. POSSIBLE ISSUE.')
            # except gp.GurobiError as exception:
            #     print('Error code: ' + str(exception))
            # except AttributeError:
            #     print('Encountered an attribute error')


        elif self.solver_flg['greedy']:
            # ip_x_sol = np.zeros(ip_c_vec.shape[0], dtype=np.uint8)
            # x_score_last = x_det_states['score'][:, -1]
            # x_last_det_ind_sorted = np.argsort(x_score_last)
            # a_mat = ip_a_mat.toarray()
            # for i in range(len(x_last_det_ind_sorted)-1, -1, -1):
            #     x_last_det_ind = x_last_det_ind_sorted[i]
            #     x_last_det_bool = x_det_ind[:, -1] == x_last_det_ind
            #     if np.any(x_last_det_bool):
            #         x_bool_ind = np.argmax(ip_c_vec[x_last_det_bool])
            #         x_sol_ind = np.flatnonzero(x_last_det_bool)[x_bool_ind]
            #         ip_x_sol[x_sol_ind] = 1
            #         constraint_rows_bool = a_mat[:, x_sol_ind] == 1
            #         constraint_cols_bool = a_mat[constraint_rows_bool, :] == 1
            #         x_bool = np.any(constraint_cols_bool, axis=0)
            #         x_score_last[x_bool] = -1
            ip_x_sol = np.zeros(ip_c_vec.shape[0], dtype=np.uint8)
            x_score_last = x_det_states['score'][:, -1]
            a_mat = ip_a_mat.toarray()
            while np.any(x_score_last > 0):
                # select highest cost from highest score detection
                x_max_last_score_ind = np.argmax(x_score_last)
                x_last_det_ind = x_det_ind[x_max_last_score_ind, -1]
                x_bool = np.all(np.stack((x_det_ind[:, -1] == x_last_det_ind,
                                          x_score_last > 0)), axis=0)
                # select highest cost overall
                # x_bool = x_score_last > 0
                x_bool_ind = np.argmax(ip_c_vec[x_bool])
                x_sol_ind = np.flatnonzero(x_bool)[x_bool_ind]
                ip_x_sol[x_sol_ind] = 1
                if not np.all(ip_a_mat @ ip_x_sol <= ip_b_vec):
                    raise ValueError('Infeasible solution found.')
                constraint_rows_bool = a_mat[:, x_sol_ind] == 1
                constraint_cols_bool = a_mat[constraint_rows_bool, :] == 1
                x_bool = np.any(constraint_cols_bool, axis=0)
                x_score_last[x_bool] = -1

        assert np.all(ip_a_mat @ ip_x_sol <= ip_b_vec)
        self.profile_end(profiler, 'solve_ip')

        return ip_x_sol


    def get_matched_indices(self, ip_x_sol, x_det_ind, tracks):
        """
        Get matched indices from IP solution.

        Args:
            ip_x_sol (np.array[Nx1]): Solution to IP problem.
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
            tracks (list[dict]): List of track dictionaries.
                sample_token (str): Sample token.
                translation (np.array): Translation in meters with shape (3).
                size (np.array): Size in meters with shape (3).
                rotation (np.array): Rotation quaternion with shape (4).
                velocity (np.array): Velocity in meters with shape (3).
                detection_name (str): Predicted class name.
                detection_score (float): Predicted class score.
                attribute_name (str): Predicted attribute name.
                ct (np.array): Center position in meters with shape (2).
                tracking (np.array): Tracking state with shape (2).
                label_preds (int): Predicted class label.
                tracking_id (int): Tracking ID.
                age (int): Age of track.
                active (int): Track is active or not.
                detection_ids (list[int]): List of track history detection IDs.

        Returns:
            matched_indices (np.array[Px2]): Detection and track indices for
                P assigned matching pairs.
        """
        profiler = self.profile_start()
        # TODO: vectorize this function
        # Get solution all detection indices and track last detection indices
        sol_det_ind_int = x_det_ind[ip_x_sol == 1, :] -1 # -1 for 0-indexing
        sol_det_ind_ext = np.array(
            [self.sw_dets['indices'][i][sol_det_ind_int[:,i]] for i in
             range(sol_det_ind_int.shape[1])]).T
        sol_det_ind_ext[sol_det_ind_int == -1] = -1
        # tracks_last_detection_ids = np.array(
        #     [track['detection_ids'][-track['age']] for track in tracks])
        # tracks_age = np.array([track['age'] for track in tracks])

        # Loop through tracks and get matched indices to current frame
        matched_indices_ext = np.zeros((0, 2), dtype=np.int16)
        if self.use_inactive_tracks:
            raise NotImplementedError

        for track_ind, track in enumerate(tracks):
            if track['age'] >= self.sw_len_curr:
                continue
            # Get track last detection ID
            track_last_det_id = track['detection_ids'][-track['age']]

            # Get solution detection IDs for given track
            sol_det_ids_ext = sol_det_ind_ext[:, -1-track['age']]

            # Get matched detection indices for given track
            matched_det_ind = np.flatnonzero(sol_det_ids_ext == track_last_det_id)
            if matched_det_ind.size == 1:
                det_ind_ext = sol_det_ind_ext[matched_det_ind[0], -1]
                matched_indices_ext = np.vstack((matched_indices_ext,
                                             np.array([det_ind_ext, track_ind])))
            elif matched_det_ind.size > 1:
                ValueError('Multiple matches found for track ID:'
                           + str(track['tracking_id']))

        # remove skipped detection and duplicate detections (keep most recent)
        matched_indices_ext = matched_indices_ext[matched_indices_ext[:, 0] != -1]
        _, unique_ind = np.unique(matched_indices_ext[:, 0], return_index=True)
        matched_indices_ext = matched_indices_ext[unique_ind]

        # sort matched indices by detection index
        sort_index_array = np.argsort(matched_indices_ext[:, 0])
        matched_indices_ext = matched_indices_ext[sort_index_array]
        self.profile_end(profiler, 'get_matched_indices')

        return matched_indices_ext


    def verify_solution(self, x_det_ind, ip_x_sol, detections, tracks, matched_indices, x_states, x_det_states):
        # checks
        sol_det_ind_int = x_det_ind[ip_x_sol == 1, :] -1 # -1 for 0-indexing
        dist_sol = x_states['dist_max'][ip_x_sol == 1]
        dist_sol_2 = x_states['neg_dist_avg_scaled'][ip_x_sol == 1]
        dist_sol_3 = x_states['skip_count'][ip_x_sol == 1]
        dist_det = np.array(
            [np.sqrt((self.sw_dets['pos_x'][-1][det_inds[-1]]
                      - self.sw_dets['pos_x'][-2][det_inds[-2]])**2 +
                     (self.sw_dets['pos_y'][-1][det_inds[-1]]
                      - self.sw_dets['pos_y'][-2][det_inds[-2]])**2)
            for det_inds in sol_det_ind_int])
        # dist_mat = np.array(
        #     [np.sqrt((tracks[match[1]]['translation'][0]
        #               - self.sw_dets['pos_x'][-1][match[0]])**2 +
        #              (tracks[match[1]]['translation'][1]
        #               - self.sw_dets['pos_y'][-1][match[0]])**2)
        #     for match in matched_indices_ext])
        det_classes = np.array(
            [detections[match[0]]['label_preds'] for match in matched_indices])
        track_classes = np.array(
            [tracks[match[1]]['label_preds'] for match in matched_indices])
        track_ids = np.array(
            [tracks[match[1]]['tracking_id'] for match in matched_indices])
        assert np.all(det_classes == track_classes)


    def debug_plot(self, ip_x_sol, x_det_ind, x_states,
                   matched_indices, tracks, sample_token, detections):
        """Plot debug visualizations."""
        profiler = self.profile_start()
        import matplotlib.pyplot as plt
        import os
        from matplotlib.lines import Line2D
        plot_ms = 1
        plot_lw = 0.5
        text_fontsize = 3
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        ax.set_xlabel("Discrete Time Step, k")
        ax.set_ylabel("Detection ID")
        ax.set_title("Multidimensional Assignment Solution")
        plot_x = np.linspace(0, self.sw_len_curr-1, self.sw_len_curr, dtype=np.int16)
        for i in range(0, x_det_ind.shape[0]):
            plot_ids = x_det_ind[i, :] > 0
            ax.plot(plot_x[plot_ids], x_det_ind[i, plot_ids]-1,
                    'ko', ms=plot_ms, lw=plot_lw)
        for i in range(0, x_det_ind.shape[0]):
            plot_ids = x_det_ind[i, :] > 0
            if ip_x_sol[i] == 1:
                ax.plot(plot_x[plot_ids], x_det_ind[i, plot_ids]-1,
                        'go-', ms=plot_ms, lw=plot_lw)
        for i in range(0, x_det_ind.shape[0]):
            plot_ids = x_det_ind[i, :] > 0
            for plot_id in np.flatnonzero(plot_ids):
                ax.text(plot_x[plot_id], x_det_ind[i, plot_id]-1,
                    str(plot_x[plot_id])+', '+str(x_det_ind[i, plot_id]-1),
                    fontsize=text_fontsize)
        out_path = os.path.join(self.work_dir, 'swmot_debug',
                            'sol_graph_'+str(self.frame_counter-1)+'-'+sample_token+'.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.cla()

        _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Matching Solution")
        for match in matched_indices:
            detection_position = detections[match[0]]['translation']
            track_position = tracks[match[1]]['translation']
            ax.plot([detection_position[0], track_position[0]],
                    [detection_position[1], track_position[1]],
                    'g', ms=plot_ms, lw=plot_lw)
            ax.scatter(detection_position[0], detection_position[1],
                    c='r', s=plot_ms)
            ax.scatter(track_position[0], track_position[1],
                    c='b', s=plot_ms)
            ax.text(detection_position[0], detection_position[1],
                    str(match[0]), fontsize=text_fontsize)
            ax.text(track_position[0], track_position[1],
                    str(match[1]), fontsize=text_fontsize)
        out_path = os.path.join(self.work_dir, 'swmot_debug',
                            'sol_match_'+str(self.frame_counter-1)+'-'+sample_token+'.png')
        handles = [
            Line2D([0], [0], label='det', color='r', marker='o', markersize=plot_ms, lw=0),
            Line2D([0], [0], label='trk', color='b', marker='o', markersize=plot_ms, lw=0)]
        ax.legend(handles=handles, loc='upper right')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.cla()

        _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Detections Solution")
        colors = plt.get_cmap('Reds')(np.linspace(0.2, 0.8, self.sw_len_curr))
        for j in range(self.sw_len_curr):
            ax.plot(self.sw_dets['pos_x'][j], self.sw_dets['pos_y'][j],
                    'o', ms=plot_ms, c=colors[j])
        for i, detection_inds in enumerate(x_det_ind):
            pos_x = []
            pos_y = []
            for f, detection_id in enumerate(detection_inds):
                if detection_id > 0:
                    pos_x.append(self.sw_dets['pos_x'][-self.sw_len_curr+f][detection_id-1])
                    pos_y.append(self.sw_dets['pos_y'][-self.sw_len_curr+f][detection_id-1])
                    ax.text(self.sw_dets['pos_x'][-self.sw_len_curr+f][detection_id-1],
                            self.sw_dets['pos_y'][-self.sw_len_curr+f][detection_id-1],
                            str(detection_id-1), fontsize=text_fontsize)
            # if ip_x_sol[i] == 0:
            #     ax.plot(pos_x, pos_y, 'gray', ms=plot_ms, lw=plot_lw)
            if ip_x_sol[i] == 1:
                ax.plot(pos_x, pos_y, 'go-', ms=plot_ms/10, lw=plot_lw)
        out_path = os.path.join(self.work_dir, 'swmot_debug',
                            'sol_det_'+str(self.frame_counter-1)+'-'+sample_token+'.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close
        self.profile_end(profiler, 'debug_plot')


    def debug_print(self, x_det_ind, detections, tracks):
        """Print debug information"""
        debug_frequency = 1
        self.prof_times.append(time.time())
        if (self.frame_counter%debug_frequency==0) and (self.frame_counter>2):
            step_time = self.prof_times[-1] - self.prof_times[-2]
            avg_time = (self.prof_times[-1] - self.prof_times[0]
                        )/(len(self.prof_times) - 1)
            process = psutil.Process()
            print("Mem Res =", round(process.memory_info().rss/1e9, 2), "GB",
                  round(process.memory_percent()), "%",
                  ", Mem Virt =", round(process.memory_info().vms/1e9, 2), "GB",
                  ", Time =", round(step_time, 2), "s",
                  ", Avg Time =", round(avg_time, 2), "s",
                  ", Vars =", x_det_ind.shape[0],
                  ", Dets =", len(detections),
                  ", Trks =", len(tracks),
                  ", SW =", self.sw_len_curr
                  )
            print("*** FRAME", self.frame_counter, "END ***")


    def profile_start(self):
        """Start profiling of function"""
        if not self.profile_flg:
            return None
        profiler = dict()
        profiler['time_start'] = time.time()
        profiler['mem_start'] = round(psutil.Process().memory_info().rss/1e6)
        return profiler


    def profile_end(self, profiler, func_name):
        """End profiling of function"""
        if not self.profile_flg:
            pass
        profiler['time_end'] = time.time()
        profiler['mem_end'] = psutil.Process().memory_info().rss/1e6
        time_usage = profiler['time_end'] - profiler['time_start']
        mem_usage = profiler['mem_end'] - profiler['mem_start']
        if time_usage > 0.01 or mem_usage > 10:
            print(">", func_name,
                "Time =", round(time_usage, 2), "s",
                ", Mem =", round(mem_usage), 
                "MB (", round(profiler['mem_end']/1000), "GB Total )")


    def assignment(self, detections, tracks, time_lag):
        """
        Performs assignment of detections to tracks.

        Args:
            detections (list[dict]): List of detection dictionaries.
                sample_token (str): Sample token.
                translation (np.array): Translation in meters with shape (3).
                size (np.array): Size in meters with shape (3).
                rotation (np.array): Rotation quaternion with shape (4).
                velocity (np.array): Velocity in meters with shape (3).
                detection_name (str): Predicted class name.
                detection_score (float): Predicted class score.
                attribute_name (str): Predicted attribute name.
                ct (np.array): Center position in meters with shape (2).
                tracking (np.array): Translation to past frame with shape (2).
                label_preds (int): Predicted class label.
            tracks (list[dict]): List of track dictionaries.
                sample_token (str): Sample token.
                translation (np.array): Translation in meters with shape (3).
                size (np.array): Size in meters with shape (3).
                rotation (np.array): Rotation quaternion with shape (4).
                velocity (np.array): Velocity in meters with shape (3).
                detection_name (str): Predicted class name.
                detection_score (float): Predicted class score.
                attribute_name (str): Predicted attribute name.
                ct (np.array): Center position in meters with shape (2).
                tracking (np.array): Tracking state with shape (2).
                label_preds (int): Predicted class label.
                tracking_id (int): Tracking ID.
                age (int): Age of track.
                active (int): Track is active or not.
                detection_ids (list[int]): List of track history detection IDs.
            time_lag (float): Timestep from last assignment to current one.

        Returns:
            matched_indices (np.array[Px2]): Detection and track indices for
                P assigned matching pairs.
        """
        self.expand_window(detections, time_lag)
        self.contract_window(tracks)
        x_det_ind = self.get_detection_to_track_map()
        x_det_ind = self.filter_map(x_det_ind)
        x_det_states = self.get_track_detection_states(x_det_ind)
        x_states = self.get_track_states(x_det_ind, x_det_states)
        ip_c_vec = self.get_cost_vec(x_states)
        x_det_ind, x_det_states, x_states, ip_c_vec = \
            self.filter_hypotheses(x_det_ind, x_det_states, x_states, ip_c_vec)
        ip_a_mat, ip_b_vec = self.get_constraints(x_det_ind)
        ip_x_sol = self.solve_ip(ip_a_mat, ip_b_vec, ip_c_vec, x_det_states,
                                 x_det_ind, x_states)
        matched_indices = self.get_matched_indices(ip_x_sol, x_det_ind, tracks)
        self.verify_solution(x_det_ind, ip_x_sol, detections, tracks, matched_indices, x_states, x_det_states)
        if self.plot_flg:
            sample_token = detections[0]['sample_token']
            self.debug_plot(ip_x_sol, x_det_ind, x_states, matched_indices,
                            tracks, sample_token, detections)
        if self.debug_flg:
            self.debug_print(x_det_ind, detections, tracks)
        return matched_indices
