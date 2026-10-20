#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure motion-aligned mask churn, a stability proxy rather than accuracy."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from visualize_results import load_results
from compare_temporal import summarize


def mask(record,key):
    result=np.zeros((240,320),np.uint8)
    for x1,y1,x2,y2 in record[key]:
        a=(round(x1*320/record['width']),round(y1*240/record['height']))
        b=(round(x2*320/record['width']),round(y2*240/record['height']))
        cv2.line(result,a,b,255,1)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=5)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output already exists')
    if args.stride<1: parser.error('Stride must be positive')
    metadata,frames=load_results(args.results)
    root=Path(metadata['input']['path'])
    cv2.setNumThreads(1)
    grid=np.stack(np.meshgrid(np.arange(320),np.arange(240)),axis=-1).astype(np.float32)
    scores={'raw':[],'temporal':[]}
    jitter={'raw':[],'temporal':[]}
    for i in range(5,len(frames),args.stride):
        previous,current=frames[i-1],frames[i]
        images=[cv2.resize(cv2.imread(str(root/f['file']),cv2.IMREAD_GRAYSCALE),(320,240),interpolation=cv2.INTER_AREA) for f in (previous,current)]
        # Backward dense flow is independent of the stabilizer's sparse LK flow.
        flow=cv2.calcOpticalFlowFarneback(images[1],images[0],None,.5,3,15,3,5,1.2,0)
        mapping=grid+flow
        for key,label in [('raw_segments','raw'),('segments','temporal')]:
            a=cv2.remap(mask(previous,key),mapping,None,cv2.INTER_NEAREST)>0
            b=mask(current,key)>0
            # Pixel distances at 320x240; compare both directions symmetrically.
            distance_a=cv2.distanceTransform((~a).astype(np.uint8),cv2.DIST_L2,3)
            distance_b=cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,3)
            distances=np.concatenate((distance_a[b],distance_b[a]))
            if len(distances):
                scores[label].append(float(np.mean(distances>1.5)))
                jitter[label].append(float(np.mean(np.minimum(distances,5))))
        if i%500==5: print(f'Analyzed pair {i}/{len(frames)}',flush=True)
    result=summarize(frames)
    result['motion_aligned_mask_churn']={key:float(np.mean(values)) for key,values in scores.items()}
    result['clipped_symmetric_mask_distance_320px']={key:float(np.mean(values)) for key,values in jitter.items()}
    result['evaluation']={'pairs':len(scores['raw']),'stride':args.stride,'resolution':[320,240],
        'method':'Backward dense Farneback warps previous line mask; symmetric distance >1.5px is unmatched. No accuracy ground truth.',
        'caveat':'Filtering away unstable but valid lines can improve this proxy. Inspect the matched side-by-side footage.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
