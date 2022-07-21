
"""Functions for evaluating results on Pascal VOC val for  AP25, AP50, AP70, AP75, ABO."""

import segm_coco_evaluate

def _do_segmentation_eval_voc(json_dataset, res_file):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = segm_coco_evaluate.SegCocoEval(json_dataset.COCO, coco_dt)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap25, ap50, ap70, ap75, abo = coco_eval.stats
    return ap25, ap50, ap70, ap75, abo




