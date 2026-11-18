"""Run the locked external COCO AP from archived evaluator JSON and local converted ground truth."""
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=["carpk","uavdt"])
    p.add_argument("converted_gt"); p.add_argument("coco_detections")
    a=p.parse_args()
    gt=COCO(a.converted_gt)
    n={"carpk":459,"uavdt":16592}[a.dataset]
    if sorted(gt.getImgIds())!=list(range(1,n+1)):
        raise ValueError("The local GT must use the archived 1-based ordered split IDs.")
    score=COCOeval(gt,gt.loadRes(a.coco_detections),"bbox")
    score.params.imgIds=sorted(gt.getImgIds())
    score.params.catIds=sorted(gt.getCatIds())
    assert len(score.params.catIds)=={"carpk":1,"uavdt":3}[a.dataset]
    score.params.maxDets=[1,10,500]
    score.evaluate(); score.accumulate()
    values=score.eval["precision"][:,:,:,0,2]
    print("AP",float(np.mean(values[values>-1])))


if __name__=="__main__": main()
