"""Private development geology review figure; never exported to researchers."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();rows=[r for r in json.loads((a.dataset/"progress.json").read_text())["records"] if r["status"]=="accepted"]
    if not rows:raise ValueError("no accepted development parents")
    fig=plt.figure(figsize=(16,4*((len(rows)+3)//4)))
    for i,row in enumerate(rows):
        parent=a.dataset/row["parent_directory"]
        meta=json.loads((parent/"private-parent.json").read_text())
        with np.load(parent/"geology.npz",allow_pickle=False) as data:truth=data["truth"]
        spacing=np.asarray(meta["spacing"]);origin=np.asarray(meta["origin"])
        vertices,faces,_,_=marching_cubes(truth.astype(float),level=.5,spacing=tuple(spacing))
        vertices+=origin+.5*spacing
        ax=fig.add_subplot((len(rows)+3)//4,4,i+1,projection="3d")
        ax.add_collection3d(Poly3DCollection(vertices[faces],facecolor="#bd783a",edgecolor="none",alpha=.9))
        extent=np.asarray(truth.shape)*spacing
        for k,axis in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):axis(origin[k],origin[k]+extent[k])
        ax.set_box_aspect(extent);ax.view_init(elev=23,azim=-58)
        ax.set_title(f"{row['parent']} / {row['family']}\ncomponents={row['component_count']}; {'noisy' if row['noisy'] else 'clean'}",fontsize=9)
        ax.tick_params(labelsize=6);ax.set_xlabel("x",fontsize=7);ax.set_ylabel("y",fontsize=7);ax.set_zlabel("z",fontsize=7)
    fig.suptitle("Development geology QA — true bodies, no method predictions\nOpen domain intersections are not padded closed",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=140);plt.close(fig)


if __name__=="__main__":main()
