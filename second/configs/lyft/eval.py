eval_detection_configs = {'lyft_cvpr_2019': {"class_range": {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "other_vehicle": 50,
    "emergency_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "animal": 40
  },
  "dist_fcn": "center_distance",
  "dist_ths": [0.5, 1.0, 2.0, 4.0],
  "dist_th_tp": 2.0,
  "min_recall": 0.1,
  "min_precision": 0.1,
  "max_boxes_per_sample": 500,
  "mean_ap_weight": 5
}}
