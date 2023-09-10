import copy
import numpy as np
from scipy import optimize
from track_utils import greedy_assignment
from swtracker import SWTracker

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car':4,
    'truck':4,
    'bus':5.5,
    'trailer':3,
    'pedestrian':1,
    'motorcycle':13,
    'bicycle':3,  
}

class PubTracker(object):
    """This is the public tracker for the NuScenes dataset."""
    def __init__(self, assigner='greedy', max_age=0, work_dir=None):
        # Assigners
        self.assigner_hungarian = False
        self.assigner_swmot = False
        self.assigner_greedy = False
        if assigner == 'greedy':
            self.assigner_greedy = True
        elif assigner == 'hungarian':
            self.assigner_hungarian = True
        elif assigner == 'swmot':
            self.assigner_swmot = True
        else:
            raise ValueError('Unknown assigner')
        assigner_count = self.assigner_hungarian + self.assigner_greedy + self.assigner_swmot
        if assigner_count > 1 or assigner_count == 0:
            raise ValueError('Only one assigner can be used')
        if self.assigner_swmot:
            self.swtracker = SWTracker(work_dir)

        # Initialize parameters
        self.max_age = max_age
        self.nuscene_cls_velocity_error = NUSCENE_CLS_VELOCITY_ERROR
        self.reset()

        # Debug info
        print("Selected assigner: ")
        if self.assigner_hungarian:
            print("Hungarian")
        elif self.assigner_greedy:
            print("Greedy")
        elif self.assigner_swmot:
            print("Sliding Window")

    def reset(self):
        """Reset internal state."""
        self.id_count = 0
        self.tracks = []
        if self.assigner_swmot:
            self.swtracker.reset()

    def step_centertrack(self, detections, time_lag):
        """Perform tracking on current frame given the detections."""
        # Extract detections and filter out classes not evaluated for tracking
        temp = []
        for det in detections:
            if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
                continue
            if det['detection_score'] < 0.0001:
                continue
            det['ct'] = np.array(det['translation'][:2])
            det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
            det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
            temp.append(det)
        detections = temp

        detections = self.swtracker.filter_detections(detections)

        # Check if detections exist
        if len(detections) == 0:
            raise ValueError('Unexpected results length') # TODO: implement if needed

        # Number of detections and tracks
        N = len(detections)
        M = len(self.tracks)

        # Extract detections [Nx2] and tracks [Mx2]
        if 'tracking' in detections[0]:
            dets = np.array(
            [ det['ct'] + det['tracking'].astype(np.float32)
            for det in detections], np.float32)
        else:
            dets = np.array([det['ct'] for det in detections], np.float32) 
        tracks = np.array(
          [pre_det['ct'] for pre_det in self.tracks], np.float32) 

        # Not first few frames
        if len(tracks) > 0:
            if self.assigner_hungarian or self.assigner_greedy or self.assigner_swmot:

                dist = (((tracks.reshape((1, -1, 2)) - \
                          dets.reshape((-1, 1, 2))) ** 2).sum(axis=2))  # N x M
                dist = np.sqrt(dist) # absolute distance in meter
                item_cat = np.array(
                    [item['label_preds'] for item in detections], np.int32) # N
                track_cat = np.array(
                    [track['label_preds'] for track in self.tracks], np.int32) # M
                max_diff = np.array(
                    [self.nuscene_cls_velocity_error[box['detection_name']] 
                     for box in detections], np.float32)
                invalid = ((dist > max_diff.reshape(N, 1)) + \
                (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
                dist = dist  + invalid * 1e18

                if self.assigner_hungarian:
                    dist[dist > 1e18] = 1e18
                    matched_indices = optimize.linear_sum_assignment(copy.deepcopy(dist))
                elif self.assigner_greedy or self.assigner_swmot:
                    matched_indices = greedy_assignment(copy.deepcopy(dist))
            if self.assigner_swmot:
                matched_indices_debug = matched_indices
                matched_indices = self.swtracker.assignment(
                    copy.deepcopy(detections), copy.deepcopy(self.tracks), 
                    time_lag)
                matches_dist = [dist[m[0], m[1]] for m in matched_indices]
        # First few frame
        else:
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)
            if self.assigner_swmot:
                self.swtracker.expand_window(copy.deepcopy(detections), time_lag)

        unmatched_dets = [d for d in range(dets.shape[0]) \
            if not (d in matched_indices[:, 0])]

        unmatched_tracks = [d for d in range(tracks.shape[0]) \
            if not (d in matched_indices[:, 1])]

        if self.assigner_hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        for m in matches:
            track = detections[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['t_age'] = 0.0
            track['active'] = self.tracks[m[1]]['active'] + 1
            track['detection_ids'] = self.tracks[m[1]]['detection_ids']
            track['detection_ids'].append(m[0])
            track['translation_history'] = self.tracks[m[1]]['translation_history']
            track['translation_history'].append(track['translation'])
            ret.append(track)

        for i in unmatched_dets:
            track = detections[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['t_age'] = 0.0
            track['active'] =  1
            track['detection_ids'] = [i]
            track['translation_history'] = [track['translation']]
            ret.append(track)

        # still store unmatched tracks if its age doesn't exceed max_age, however,
        # we shouldn't output the object in current frame
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['t_age'] += time_lag
                track['active'] = 0
                track['detection_ids'].append(-1)
                ct = track['ct']

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1 # move forward
                    track['ct'] = ct + offset
                ret.append(track)

        self.tracks = ret
        return ret, detections
