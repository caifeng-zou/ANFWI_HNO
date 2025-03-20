import numpy as np
import matplotlib.pyplot as plt

def plot_fwi_result(vp_init, vp_inv, vs_init, vs_inv, xrange, yrange, nx, ny, 
                    name1="Initial $\mathregular{V_P}$", 
                    name2="Inverted $\mathregular{V_P}$",
                    name3="Initial $\mathregular{V_S}$",
                    name4="Inverted $\mathregular{V_S}$",
                    figsize=(20, 8), hspace=0.4, fraction=0.018, pad=0.02, savepath=None, xs=None, zs=None, y0=None, y1=None):
    x, y = np.linspace(0, xrange/1000, nx), np.linspace(0, yrange/1000, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    vp_init = vp_init / 1000
    vp_inv = vp_inv / 1000   
    vs_init = vs_init / 1000
    vs_inv = vs_inv / 1000

    vpmax = 4.4 * 1.732
    vpmin = 1.6 * 1.732
    vsmax = 4.4
    vsmin = 1.6
    
    fig, axarr = plt.subplots(2, 2, figsize=figsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18, hspace=hspace)
    fig.facecolor = 'white'
    plt.rcParams['axes.facecolor'] = 'w'
    for ax, data, name, vmin, vmax in zip(axarr.flatten(), 
                              [vp_init, vp_inv, vs_init, vs_inv],
                              [name1,
                               name2,
                               name3,
                               name4],
                               [vpmin, vpmin, vsmin, vsmin],
                               [vpmax, vpmax, vsmax, vsmax]):
        if data is None:
            ax.axis('off')
            continue
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('bottom')
        ax.invert_yaxis()
        ax.set_title(name, fontsize=18, fontweight='bold')
        ax.tick_params(labelsize=15, width=2)
        im1 = ax.pcolormesh(xx, yy, data, vmin=vmin, vmax=vmax, cmap='Spectral', rasterized=True, shading='nearest')
        clb1 = plt.colorbar(im1, ax=ax, fraction=fraction, pad=pad)
        clb1.ax.tick_params(labelsize=15, width=2)
        for label in clb1.ax.get_yticklabels():
            label.set_weight('bold')
        clb1.outline.set_linewidth(2)
        clb1.set_label('km/s', fontsize=15, fontweight='bold')
        clb1.ax.yaxis.set_label_position('left')  
        clb1.ax.yaxis.label.set_rotation(0) 
        clb1.ax.yaxis.set_label_coords(1.5, 1)   
        ax.set_xlabel("X (km)", fontsize=15, fontweight='bold')
        ax.set_ylabel("Z (km)", fontsize=15, fontweight='bold')
        ax.set_ylim(y1, y0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_weight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
        if xs is not None and zs is not None and 'Inverted' in name:
            ax.plot(xs, zs, 'k', linewidth=2, linestyle='--', label='Villa et al. 2023')
            ax.legend(fontsize=15)
    
    if savepath is not None:
        plt.savefig(savepath, dpi=600, bbox_inches='tight', format='pdf')