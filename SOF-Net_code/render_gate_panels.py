import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',nargs='+',required=True,help='Gate NPZ files selected by real patch IDs')
    p.add_argument('--output',required=True)
    p.add_argument('--rgb-bands',nargs=3,type=int,required=True,help='Zero-based RGB band indices in stored optical input')
    a=p.parse_args()
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':10})
    fig,axes=plt.subplots(len(a.inputs),6,figsize=(15,2.6*len(a.inputs)),squeeze=False,layout='constrained')
    cmap=ListedColormap(['black','#dcebc8','#54892c','#ffdf00','#ed2525','#167bc0'])
    for row,file in enumerate(a.inputs):
        metadata=json.loads(Path(file).with_suffix('.json').read_text(encoding='utf-8'))
        if metadata.get('source')!='actual forward-pass tensor' or not metadata.get('checkpoint_sha256'):
            raise ValueError('Missing checkpoint provenance')
        with np.load(file,allow_pickle=False) as d:
            rgb=d['optical'][a.rgb_bands].transpose(1,2,0)
            panels=[rgb,d['prior'].squeeze(),d['quality'].squeeze(),d['G_mean'],d['reference'],d['prediction']]
            for col,panel in enumerate(panels):
                if col==0:
                    axes[row,col].imshow(np.clip(panel,0,1))
                elif col<4:
                    im=axes[row,col].imshow(panel,cmap='viridis',vmin=0,vmax=1)
                else:
                    panel=np.ma.masked_where((panel==255)|~d['valid'].astype(bool),panel)
                    axes[row,col].imshow(panel,cmap=cmap,vmin=0,vmax=5,interpolation='nearest')
                axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            axes[row,0].set_ylabel(metadata['patch_id'])
    for ax,title in zip(axes[0],['GF-7 (display stretch)','SAR prior','Quality M','SAR gate G','Reference mask','Predicted mask']):
        ax.set_title(title)
    fig.colorbar(im,ax=axes[:,1:4].ravel().tolist(),shrink=.7,label='Fixed scale 0–1; larger G = more SAR weight')
    fig.suptitle('Fig. S3-1. Checkpoint-derived quality-aware gate activations')
    destination=Path(a.output); destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=300)
    plt.close(fig)


if __name__=='__main__':
    main()
