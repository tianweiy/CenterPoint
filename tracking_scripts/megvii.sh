#!/bin/bash

python tools/tracking/pub_test.py --work_dir work_dirs/megvii_track  --checkpoint work_dirs/megvii_track/megvii_val.json  --max_age 3
