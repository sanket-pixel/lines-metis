#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare raw and stabilized cached endpoints with identical rendering."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from visualize_results import BG, FG, MUTED, CYAN, font, load_results


def summarize(frames):
    raw = np.array([len(f['raw_segments']) for f in frames])
    stable = np.array([len(f['segments']) for f in frames])
    temporal = np.array([f['temporal_ms'] for f in frames])
    return {'frames': len(frames), 'mean_segments': {'raw': float(raw.mean()), 'temporal': float(stable.mean())},
            'mean_absolute_count_change': {'raw': float(np.abs(np.diff(raw)).mean()), 'temporal': float(np.abs(np.diff(stable)).mean())},
            'temporal_ms': {'mean': float(temporal.mean()), 'p95': float(np.percentile(temporal,95))},
            'note': 'Count change measures population stability, not accuracy or geometric jitter. Fewer displayed lines can lower it.'}


class Comparison:
    def __init__(self, metadata, frames):
        if 'temporal' not in metadata or any('raw_segments' not in f for f in frames):
            raise ValueError('Expected stabilized export with raw_segments')
        self.frames = frames
        self.root = Path(metadata['input']['path'])
        self.stats = summarize(frames)
        self.base = Image.new('RGB',(1920,1080),BG)
        d = ImageDraw.Draw(self.base)
        d.text((48,30),'lines / metis',font=font(40, bold=True),fill=FG)
        d.text((48,85),'TEMPORAL CONSISTENCY  /  MOTION-COMPENSATED LINE SEGMENTS',font=font(20),fill=MUTED)
        d.text((1520,50),'C++  /  CAUSAL FILTER',font=font(22),fill=CYAN)
        d.line((48,126,1872,126),fill=(45,59,70))
        for x,title,subtitle in [(48,'01  /  FRAME BY FRAME','Balanced detections · frozen baseline'),(976,'02  /  TEMPORAL LAYER','Motion prediction · confirmation · short gap recovery')]:
            d.text((x,156),title,font=font(25),fill=FG)
            d.text((x,196),subtitle,font=font(20),fill=MUTED)
        d.text((48,966),'Same images. Same saved model outputs. Identical line styling.',font=font(22),fill=FG)
        d.text((48,1002),'No retraining or future frames. Temporal filtering may suppress short-lived valid lines.',font=font(20),fill=MUTED)
        d.text((48,1044),'TUM RGB-D · Freiburg 2 pioneer SLAM · J. Sturm et al. · CC BY 4.0',font=font(18),fill=MUTED)
        d.text((1330,1044),'cvg.cit.tum.de/data/datasets/rgbd-dataset',font=font(18),fill=MUTED)

    def render(self,index):
        record=self.frames[index]
        image=Image.open(self.root/record['file']).convert('RGB').resize((896,672),Image.Resampling.LANCZOS)
        image=Image.blend(image,Image.new('RGB',image.size,(13,26,32)),.12)
        panels=[]
        for key in ('raw_segments','segments'):
            panel=np.array(image)
            mask=np.zeros(panel.shape[:2],np.uint8)
            for x1,y1,x2,y2 in record[key]:
                a=(round(x1*896/record['width']),round(y1*672/record['height']))
                b=(round(x2*896/record['width']),round(y2*672/record['height']))
                cv2.line(mask,a,b,255,2,cv2.LINE_AA)
            visible=mask>0
            alpha=mask[visible,None]/255
            panel[visible]=(panel[visible]*(1-alpha)+np.array(CYAN)*alpha).astype(np.uint8)
            panels.append(Image.fromarray(panel))
        canvas=self.base.copy()
        canvas.paste(panels[0],(48,244));canvas.paste(panels[1],(976,244))
        d=ImageDraw.Draw(canvas)
        for x,key in ((48,'raw_segments'),(976,'segments')):
            d.text((x,928),f'{len(record[key]):3d} segments',font=font(22,mono=True),fill=CYAN)
        d.text((1240,928),f"Temporal overhead: {self.stats['temporal_ms']['mean']:.1f} ms mean",font=font(22),fill=MUTED)
        elapsed=record['timestamp_seconds']-self.frames[0]['timestamp_seconds']
        d.text((1550,986),f'{index+1:04d} / {len(self.frames):04d}',font=font(22,mono=True),fill=MUTED)
        d.text((1550,1012),f'{elapsed:6.2f}s source',font=font(18,mono=True),fill=MUTED)
        d.line((48,1035,1872,1035),fill=(45,59,70),width=2)
        d.line((48,1035,48+1824*(index+1)/len(self.frames),1035),fill=CYAN,width=2)
        return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preview',type=Path)
    parser.add_argument('--preview-frame',type=int,default=600)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--frames',type=int)
    parser.add_argument('--fps',type=float,default=30)
    args=parser.parse_args()
    metadata,frames=load_results(args.results)
    if args.fps<=0 or not 0<=args.start<len(frames) or (args.frames is not None and args.frames<=0):
        parser.error('Invalid frame range or FPS')
    for path in (args.output,args.preview):
        if path and path.exists(): parser.error(f'Refusing to overwrite {path}')
        if path: path.parent.mkdir(parents=True,exist_ok=True)
    renderer=Comparison(metadata,frames)
    if args.preview:
        if not 0<=args.preview_frame<len(frames): parser.error('Preview frame outside export')
        renderer.render(args.preview_frame).save(args.preview)
    if not args.output: return
    selected=range(args.start,min(len(frames),args.start+args.frames) if args.frames else len(frames))
    command=['ffmpeg','-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for count,index in enumerate(selected):
            process.stdin.write(renderer.render(index).tobytes())
            if count%100==0: print(f'Rendered {count+1}/{len(selected)}',flush=True)
        process.stdin.close()
        if process.wait(): raise RuntimeError('FFmpeg encoding failed')
    finally:
        if process.poll() is None:
            process.kill();process.wait()
    stats=renderer.stats|{'source':str(args.results),'start':args.start,'rendered_frames':len(selected),'playback_fps':args.fps}
    args.output.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__=='__main__':
    main()
