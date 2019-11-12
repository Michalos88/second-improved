import pickle

import fire
import ray
import numpy as np
from second.data.lyft_eval import eval_main


def get_score(gt_path, preds_path, cpus=2):

    with open(gt_path, "rb") as  f:
        gt_annos = pickle.load(f)

    with open(preds_path, "rb") as  f:
        preds = pickle.load(f)

    ray.init(memory=10*10**9)
    mAPs = [eval_main.remote(gt_annos,
                             preds,
                             round(threshold, 3))
            for threshold in np.arange(0.5, 1.0, 0.05)]

    print("Final Score = ", np.mean(ray.get(mAPs)))


if __name__ == '__main__':
    fire.Fire(get_score)
