import fire
import numpy as np
import ray
from lyft_dataset_sdk.eval.detection.mAP_evaluation\
        import get_average_precisions, get_class_names


@ray.remote
def eval_main(gt, predictions, iou_threshold):

    class_names = get_class_names(gt)

    average_precisions = get_average_precisions(gt,
                                                predictions,
                                                class_names,
                                                iou_threshold)

    mAP = np.mean(average_precisions)
    print("Average per class mean average precision @",
          iou_threshold,
          "  = ", mAP)

    for class_id in sorted(
            list(zip(class_names, average_precisions.flatten().tolist()))):
        print(class_id)

    return mAP


if __name__ == "__main__":
    fire.Fire(eval_main)
