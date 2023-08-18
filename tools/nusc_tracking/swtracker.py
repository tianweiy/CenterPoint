import time
import numpy as np
import scipy.sparse as sp
try:
    import gurobipy as gp
except ImportError:
    pass

class SWTracker():
    """Sliding window tracker class."""
    def __init__(self, work_dir=None):
        self.cost_flg = {"dist_max": False, "dist_avg": False, "dist_sum": False,
                         "llr": True, "learned": False}
        assert sum(self.cost_flg.values()) == 1
        self.solver_flg = {"gurobi": False, "greedy": True}
        assert sum(self.solver_flg.values()) == 1
        self.sw_len_max = 3
        assert self.sw_len_max >= 2
        self.plot_flg = False
        self.work_dir = work_dir
        self.use_inactive_tracks = False
        self.frame_counter = 0

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
                        "num_dets": np.array([], dtype=np.int32), "indices": [],
                        }
        self.sw_len_curr = 0
        self.ip_sol = {}


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
        self.sw_dets['delta_t'].append(time_lag)
        self.sw_dets['pos_x'].append(
            np.array([det['translation'][0] for det in detections], np.float32))
        self.sw_dets['pos_y'].append(
            np.array([det['translation'][1] for det in detections], np.float32))
        self.sw_dets['vel_x'].append(
            np.array([det['velocity'][0] for det in detections], np.float32))
        self.sw_dets['vel_y'].append(
            np.array([det['velocity'][1] for det in detections], np.float32))
        self.sw_dets['class_id'].append(
            np.array([det['label_preds'] for det in detections], np.int32))
        self.sw_dets['score'].append(
            np.array([det['detection_score'] for det in detections], np.float32))
        self.sw_dets['max_vel'].append(
            np.array([self.class_vel_limit[det['detection_name']]
                for det in detections], np.float32))
        self.sw_dets['indices'].append(np.arange(len(detections) + 1))

        self.sw_dets['num_dets'] = np.append(
            self.sw_dets['num_dets'], len(detections))
        if self.sw_len_curr < self.sw_len_max:
            self.sw_len_curr += 1
        self.frame_counter += 1


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
        # Contract window # TODO: most efficient way to pop list and aray?
        if self.sw_len_curr > self.sw_len_max:
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

        # Replace detections with tracks at beginning of window
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


    def get_detection_to_track_map(self):
        '''
        Get mapping of detection indices to track decision variables.

        Returns:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
        '''
        # TODO: reuse calculation from previous iteration
        # Create tracks with all permuations of detection indices
        sw_num_dets = self.sw_dets['num_dets'][-self.sw_len_curr:]
        ip_len_x = int(np.prod(sw_num_dets + 1))
        x_det_ind = np.zeros((ip_len_x, self.sw_len_curr), dtype=np.int64)
        for iter_frame in range(-self.sw_len_curr, -1):
            x_det_ind[:, iter_frame] = np.tile(np.repeat(
                self.sw_dets['indices'][iter_frame],
                np.prod(sw_num_dets[iter_frame+1:] + 1)),
                np.prod(sw_num_dets[:iter_frame] + 1))
        x_det_ind[:, -1] = np.tile(self.sw_dets['indices'][-1],
            np.prod(sw_num_dets[:-1] + 1))

        # Filter out tracks with less than 2 detections
        # Detection index = 0 means no detection (frame skipped)
        x_det_ind = x_det_ind[np.sum(x_det_ind>0, axis=1) >= 2, :]
        ip_len_x = int(np.prod(sw_num_dets + 1) - np.sum(sw_num_dets) - 1)
        assert x_det_ind.shape[0] == ip_len_x

        return x_det_ind


    def get_track_detection_states(self, x_det_ind):
        """
        Get states for the detections of each track decision variable.

        Args:
            x_det_ind (np.array[NxT]): Array of detection indices for
                N decision variables and T frames.
        
        Returns:
            x_det_states (dict[np.array[NxT]]): dictionary of detection states, 
                each state is an array for N decision variables and T frames.
        """
        x_det_states = {}
        x_det_states['class_id'] = np.ones(x_det_ind.shape)*-1
        x_det_states['pos_x'] = np.zeros(x_det_ind.shape)
        x_det_states['pos_y'] = np.zeros(x_det_ind.shape)
        x_det_states['vel_x'] = np.zeros(x_det_ind.shape)
        x_det_states['vel_y'] = np.zeros(x_det_ind.shape)
        x_det_states['max_vel'] = np.zeros(x_det_ind.shape)
        x_det_states['score'] = np.zeros(x_det_ind.shape)

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
        x_class_vec = -1*np.ones(x_det_ind.shape[0], dtype=int)
        x_class_vec[x_cond_1class_det] = x_class_sorted[x_cond_1class_det, 0]
        x_class_vec[x_cond_2class_nodet] = x_class_sorted[x_cond_2class_nodet, 1]
        x_states['class_id'] = x_class_vec # -1: multi-class, 0: class 0, 1: class 1

        # Get distance limit vector
        class_vel_limits = list(self.class_vel_limit.values())
        class_vel_limits.append(0)
        class_dist_limits = np.array(class_vel_limits)*0.5
        x_dist_limit_vec = class_dist_limits[x_class_vec]
        x_states['dist_limit'] = x_dist_limit_vec

        map_shape = np.shape(x_det_ind)
        num_row = np.size(x_det_ind, 0)
        num_col = np.size(x_det_ind, 1)
        x_states['dist_sum'] = np.zeros(num_row)
        x_states['dist_sum_scaled'] = np.zeros(num_row)
        x_states['dist_max'] = np.zeros(num_row)
        x_states['dist_max_scaled'] = np.zeros(num_row)
        x_states['frames'] = np.zeros(num_row)
        # loop over all possible pairs of detections across window
        for col_a in range(0, num_col-1):
            for col_b in range(col_a+1, num_col):
                # Select rows with both detections and none in between
                map_bool = np.zeros(map_shape)
                map_cond = np.zeros(map_shape)
                map_bool[:, col_a:col_b+1] = x_det_ind[:, col_a:col_b+1] > 0
                map_cond[:, col_a] = 1
                map_cond[:, col_b] = 1
                row_bool = np.all(map_bool == map_cond, axis=1)

                # Calculate time step and number of frames between detections
                sw_delta_t = [self.sw_dets['delta_t'][col_id] for col_id in
                               range(-self.sw_len_curr, 0)]
                delta_t = np.sum(sw_delta_t[col_a+1:col_b+1])
                pair_frames = (col_b-col_a)*np.ones((num_row))[row_bool]
                x_states['frames'][row_bool] = x_states['frames'][row_bool]\
                    + pair_frames

                # Compute distance between detections for given pair of frames
                pair_dist_x = x_det_states['pos_x'][row_bool, col_b] - \
                    x_det_states['vel_x'][row_bool, col_b] * delta_t - \
                    x_det_states['pos_x'][row_bool, col_a]
                pair_dist_y = x_det_states['pos_y'][row_bool, col_b] - \
                    x_det_states['vel_y'][row_bool, col_b] * delta_t - \
                    x_det_states['pos_y'][row_bool, col_a]
                pair_dist = np.sqrt(pair_dist_x**2 + pair_dist_y**2)
                scale_max = 15
                pair_dist_scaled = (scale_max - pair_dist)/scale_max

                # Compute running total distance, max distance, scores
                x_states['dist_sum'][row_bool] = x_states['dist_sum'][row_bool]\
                    + pair_dist
                x_states['dist_max'][row_bool] = np.amax(np.stack(
                    [x_states['dist_max'][row_bool], pair_dist/pair_frames],
                    axis=1), axis=1)
                x_states['dist_sum_scaled'][row_bool] = \
                    x_states['dist_sum_scaled'][row_bool] + pair_dist_scaled
                x_states['dist_max_scaled'][row_bool] = np.amax(np.stack(
                    [x_states['dist_max_scaled'][row_bool],
                     pair_dist_scaled/pair_frames], axis=1), axis=1)

        x_states['dist_avg'] = x_states['dist_sum']/x_states['frames']
        x_states['dist_avg_scaled'] = x_states['dist_sum_scaled']/x_states['frames']
        x_states['score_sum'] = np.sum(x_det_states['score'], axis=1)
        x_states['score_avg'] = x_states['score_sum']/(x_states['frames']+1)
        x_states['score_min'] = np.min(x_det_states['score'],
            where=x_det_states['score']!=0, initial = 1, axis=1)
        x_states['skip_count'] = np.sum(x_det_ind == 0, axis=1)

        return x_states


    def filter_tracks(self, x_det_ind, x_det_states, x_states):
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
        # Condition for tracks with one class and detections in distance limit
        x_dist_cond = x_states['dist_max'] < x_states['dist_limit']
        x_class_cond = x_states['one_class_bool']
        x_score_cond = x_states['score_min'] > 0.0001
        x_cond = np.all(np.stack([x_dist_cond, x_class_cond, x_score_cond], 
                                 axis=1), axis=1)

        # Filter out decision variables based on condition
        x_states = {key: val[x_cond] for key, val in x_states.items()}
        x_det_ind = x_det_ind[x_cond, :]
        x_det_states = {key: val[x_cond, :] for key, val in x_det_states.items()}

        return x_det_ind, x_det_states, x_states


    def get_cost_vec(self, x_states):
        """
        Get cost vector for detections in format of x.

        Args:
            x_states (dict[np.array[N]]): dictionary of track states,
                each state is an array for N decision variables.
        
        Returns:
            cost_vec (np.array[Nx1]): Cost vector for N decision variables.
        """
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
            cost_vec = x_states['dist_sum_scaled']
        # Probabilistic log likelihood ratio
        if self.cost_flg['llr']:
            dist_cost = x_states['dist_avg_scaled']
            signal_cost = x_states['score_avg']*0
            skip_cost = -x_states['skip_count']*1
            cost_vec = dist_cost + signal_cost + skip_cost
        # Learned affinity
        elif self.cost_flg['learned']:
            raise NotImplementedError('Learned affinity not implemented yet.')
        x_states['cost_vec'] = cost_vec
        return cost_vec


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

        return ip_a_mat, ip_b_vec


    def solve_ip(self, ip_a_mat, ip_b_vec, ip_c_vec, x_det_states, x_det_ind, debug_flg=False):
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

        elif self.solver_flg['greedy']:
            ip_x_sol = np.zeros(ip_c_vec.shape[0])
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
                if np.all(ip_a_mat @ ip_x_sol <= ip_b_vec) == False:
                    raise ValueError('Infeasible solution found.')
                constraint_rows_bool = a_mat[:, x_sol_ind] == 1
                constraint_cols_bool = a_mat[constraint_rows_bool, :] == 1
                x_bool = np.any(constraint_cols_bool, axis=0)
                x_score_last[x_bool] = -1

        assert np.all(ip_a_mat @ ip_x_sol <= ip_b_vec)

        return ip_x_sol


    def get_matched_indices(self, ip_x_sol, x_det_ind, tracks, x_states):
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
        # TODO: vectorize this function
        # Get solution all detection indices and track last detection indices
        sol_det_ind = x_det_ind[ip_x_sol == 1, :] -1 # -1 for 0-indexing
        # tracks_last_detection_ids = np.array(
        #     [track['detection_ids'][-track['age']] for track in tracks])
        # tracks_age = np.array([track['age'] for track in tracks])

        # Loop through tracks and get matched indices to current frame
        matched_indices = np.zeros((0, 2), dtype=int)
        if self.use_inactive_tracks:
            raise NotImplementedError

        for track_ind, track in enumerate(tracks):
            if track['age'] >= self.sw_len_curr:
                continue
            # Get track last detection ID
            track_last_det_id = track['detection_ids'][-track['age']]

            # Get solution detection IDs for given track
            sol_det_ids = sol_det_ind[:, -1-track['age']]

            # Get matched detection indices for given track
            matched_det_ind = np.flatnonzero(sol_det_ids == track_last_det_id)
            if matched_det_ind.size == 1:
                det_ind = sol_det_ind[matched_det_ind[0], -1]
                matched_indices = np.vstack((matched_indices,
                                             np.array([det_ind, track_ind])))
            elif matched_det_ind.size > 1:
                ValueError('Multiple matches found for track ID:'
                           + str(track['tracking_id']))

        # # Loop through past frames and get matched indices to current frame
        # matched_indices = np.zeros((0, 2), dtype=int)
        # for age in np.sort(np.unique(tracks_age)):
        #     if age > self.sw_len_curr-1:
        #         break

        #     # Get track IDs and detection IDs for given age
        #     tracks_age_indices = np.where(tracks_age == age)
        #     tracks_age_det_ids = tracks_last_detection_ids[tracks_age_indices]

        #     # Get solution detection IDs for given age
        #     sol_age_det_ids = sol_det_ind[:, -1-age]
        #     sol_cur_det_ids = sol_det_ind[:, -1]

        #     # Match detection indices of tracks and solution for given age
        #     _, track_ind, sol_ind = np.intersect1d(
        #         tracks_age_det_ids, sol_age_det_ids, return_indices=True)

        #     # Concatenate track and current detection indices
        #     matched_age_track_ids = tracks_age_det_ids[track_ind]
        #     matched_age_det_ids = sol_cur_det_ids[sol_ind]
        #     matched_age_indices = np.stack((matched_age_det_ids,
        #                                     matched_age_track_ids), axis=1)
        #     matched_indices = np.concatenate(
        #         (matched_indices, matched_age_indices), axis=0)

        # remove skipped detection and duplicate tracks (keep most recent)
        matched_indices = matched_indices[matched_indices[:, 0] != -1]
        _, unique_ind = np.unique(matched_indices[:, 1], return_index=True)
        matched_indices = matched_indices[unique_ind]


        # checks
        dist_sol = x_states['dist_max'][ip_x_sol == 1]
        dist_sol_2 = x_states['dist_avg_scaled'][ip_x_sol == 1]
        dist_sol_3 = x_states['skip_count'][ip_x_sol == 1]
        dist_det = np.array(
            [np.sqrt((self.sw_dets['pos_x'][-1][det_inds[-1]]
                      - self.sw_dets['pos_x'][-2][det_inds[-2]])**2 +
                     (self.sw_dets['pos_y'][-1][det_inds[-1]]
                      - self.sw_dets['pos_y'][-2][det_inds[-2]])**2)
            for det_inds in sol_det_ind])
        dist_mat = np.array(
            [np.sqrt((tracks[match[1]]['translation'][0]
                      - self.sw_dets['pos_x'][-1][match[0]])**2 +
                     (tracks[match[1]]['translation'][1]
                      - self.sw_dets['pos_y'][-1][match[0]])**2)
            for match in matched_indices])

        # sort matched indices by detection index
        sort_index_array = np.argsort(matched_indices[:, 0])
        matched_indices = matched_indices[sort_index_array]

        return matched_indices


    def debug_plot(self, ip_x_sol, x_det_ind, x_states, matched_indices, tracks, sample_token, detections):
        import matplotlib.pyplot as plt
        import os
        plot_ms = 1
        plot_lw = 0.5
        text_fontsize = 3
        _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_xlabel("Discrete Time Step, k")
        ax.set_ylabel("Detection ID")
        ax.set_title("Multidimensional Assignment Solution")
        plot_x = np.linspace(0, self.sw_len_curr-1, self.sw_len_curr, dtype=int)
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
        plt.close

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
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close

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
        import sys
        print(sys.getsizeof(self))
        print(sys.getsizeof(self.sw_dets))
        sample_token = detections[0]['sample_token']
        self.expand_window(detections, time_lag)
        self.contract_window(tracks)
        x_det_ind = self.get_detection_to_track_map()
        x_det_states = self.get_track_detection_states(x_det_ind)
        x_states = self.get_track_states(x_det_ind, x_det_states)
        x_det_ind, x_det_states, x_states = \
            self.filter_tracks(x_det_ind, x_det_states, x_states)
        ip_c_vec = self.get_cost_vec(x_states)
        ip_a_mat, ip_b_vec = self.get_constraints(x_det_ind)
        ip_x_sol = self.solve_ip(ip_a_mat, ip_b_vec, ip_c_vec, x_det_states, x_det_ind)
        matched_indices = self.get_matched_indices(ip_x_sol, x_det_ind, tracks, x_states)
        if self.plot_flg:
            self.debug_plot(ip_x_sol, x_det_ind, x_states, matched_indices, tracks, sample_token, detections)
        return matched_indices
