import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import matplotlib.gridspec as gridspec

cmap=plt.get_cmap('cool')

if __name__ == '__main__':

    fig = plt.figure(tight_layout=True, figsize=(12,3.5))
    gs = gridspec.GridSpec(1, 2)

    dims_to_consider = [1024, 1280, 1408, 1664, 2048, 4096]
    batch_size_for_plot1 = 32768
    batch_sizes_for_plot2 = [2**14, 2**15, 2**16, 2**17]
    dims_to_xtick = [1024, 2048, 4096]
    logscale_plot1 = True

    ax = fig.add_subplot(gs[0, 0])

    # TODO: change this to what you want.
    rdf_ffb = pd.read_json('speed_benchmark_ffb/info.jsonl', lines=True)
    rdf_bnb = pd.read_json('speed_benchmark_bnb/info.jsonl', lines=True)
    df_ffb = rdf_ffb[rdf_ffb.batch_size == batch_size_for_plot1]
    df_bnb = rdf_bnb[rdf_bnb.batch_size == batch_size_for_plot1]

    plot_desc = {
        "bnb": [
            ('standard_gx+standard_gw+standard_fwd', 's', '-', 'b', f'Standard fp16 (sum of parts) bnb'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'b', f'SwitchBack int8 (sum of parts) bnb'),

            ('standard_fwd', '^', '--', 'b', f'Matmul XW (standard) bnb'),
            ('standard_gw', '^', '-.', 'b', f'Matmul GW (standard) bnb'),
            ('standard_gx', '^', ':', 'b', f'Matmul GX (both) bnb'),

            ('global_fwd', '^', '--', 'b', f'Int8 Matmul XW (switchback) bnb'),
            ('global_bwd', '^', '-.', 'b', f'Int8 Matmul GW (switchback) bnb'),
            
            ('x_quantize_rowwise', 'P', '--', 'b', f'Quantize rowwise X (switchback) bnb'),
            ('g_quantize_rowwise', 'P', '-.', 'b', f'Quantize rowwise G (switchback) bnb'),
            ('w_quantize_global', '.', '--', 'b', f'Quatnize global W (switchback) bnb'),
            ('w_quantize_global_transpose', '.', '-.', 'b', f'Quantize gloabl and\ntranspose W (switchback) bnb'),
        ],
        "ffb": [
            ('standard_gx+standard_gw+standard_fwd', 's', '-', 'r', f'Standard fp16 (sum of parts) ffb'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'r', f'SwitchBack int8 (sum of parts) ffb'),

            ('standard_fwd', '^', '--', 'r', f'Matmul XW (standard) ffb'),
            ('standard_gw', '^', '-.', 'r', f'Matmul GW (standard) ffb'),
            ('standard_gx', '^', ':', 'r', f'Matmul GX (both) ffb'),

            ('global_fwd', '^', '--', 'r', f'Int8 Matmul XW (switchback) ffb'),
            ('global_bwd', '^', '-.', 'r', f'Int8 Matmul GW (switchback) ffb'),
            
            ('x_quantize_rowwise', 'P', '--', 'r', f'Quantize rowwise X (switchback) ffb'),
            ('g_quantize_rowwise', 'P', '-.', 'r', f'Quantize rowwise G (switchback) ffb'),
            ('w_quantize_global', '.', '--', 'r', f'Quatnize global W (switchback) ffb'),
            ('w_quantize_global_transpose', '.', '-.', 'r', f'Quantize gloabl and\ntranspose W (switchback) ffb'),
        ]
    }
    # first plot the time occupied by different operations
    for exp in ['bnb', 'ffb']:
        df = df_ffb if exp == 'ffb' else df_bnb
        for k, marker, ls, color, name in plot_desc[exp]:
            xs = []
            ys = []
            for embed_dim in dims_to_consider:
                # average over dim -> 4*dim and 4*dim -> dim
                df_ = df[df.dim_in == embed_dim]
                df_ = df_[df_.dim_out == embed_dim * 4]
                xs.append(embed_dim)
                y_ = 0
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                df_ = df[df.dim_in == embed_dim * 4]
                df_ = df_[df_.dim_out == embed_dim]
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                ys.append(y_ * 0.5)

            
            ax.plot(xs, ys, color=color, label=name, marker=marker, markersize=5 if marker=='s' else 5, linestyle=ls, linewidth=2 if '+' in k else 1.)


    ax.set_xlabel('dim', fontsize=13)
    ax.set_ylabel('time (ms)', fontsize=13)

    ax.grid()

    ax.set_xscale('log')
    if logscale_plot1:
        ax.set_yscale('log')
    
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks(dims_to_xtick)
    ax.set_xticklabels(dims_to_xtick)
    ax.set_xticks([], minor=True)

    # leg = ax.legend(loc='upper center', bbox_to_anchor=(-0.64,  1.), ncol=1, fontsize=10)
    # leg.get_texts()[0].set_fontweight('bold')
    # leg.get_texts()[1].set_fontweight('bold')
    plt.subplots_adjust(left=0.1)
    ax.set_title('  Linear layer, batch * sequence length = 32k', fontsize=10, loc='left', y=1.05, pad=-20)


    ax = fig.add_subplot(gs[0, 1])

    plot_desc_2 = {
        "ffb": [
            ('standard_gx+standard_gw+standard_fwd', 's', '-', 'r', f'Standard fp16 (sum of parts) ffb'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'r', f'SwitchBack int8 (sum of parts) ffb'),
        ],
        "bnb": [
            ('standard_gx+standard_gw+standard_fwd', 's', '-', 'b', f'Standard fp16 (sum of parts) bnb'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'b', f'SwitchBack int8 (sum of parts) bnb'),
        ]
    }

    # now plot the % speedup for different batch sizes
    for i, exp in enumerate(['bnb', 'ffb']):
        rdf = rdf_ffb if exp == 'ffb' else rdf_bnb
        for j, batch_size in enumerate(batch_sizes_for_plot2):
            all_xs, all_ys = [], []
            for k, marker, ls, color, name in plot_desc_2[exp]:
            
                xs, ys = [], []
                df = rdf[rdf.batch_size == batch_size]
                for embed_dim in dims_to_consider:
                    df_ = df[df.dim_in == embed_dim]
                    df_ = df_[df_.dim_out == embed_dim * 4]
                    xs.append(embed_dim)
                    y_ = 0
                    for k_ in k.split('+'):
                        y_ += df_[k_].values[0]
                    df_ = df[df.dim_in == embed_dim * 4]
                    df_ = df_[df_.dim_out == embed_dim]
                    for k_ in k.split('+'):
                        y_ += df_[k_].values[0]
                    ys.append(y_ * 0.5)
                all_xs.append(xs)
                all_ys.append(ys)

            color = cmap((j + i * len(batch_sizes_for_plot2)) * 0.25)
            real_ys = [-((all_ys[1][i] - all_ys[0][i]) / all_ys[0][i]) * 100 for i in range(len(all_ys[0]))]
            markers = ['^', 'v', 'P', 'o']
            ax.plot(all_xs[0], real_ys, color=color, label=f'batch * sequence length = {batch_size}', marker=markers[j], markersize=5 if marker=='s' else 5)

    # ax.legend()
    ax.set_xlabel('dim', fontsize=13)
    ax.set_xscale('log')
    ax.grid()
    ax.set_ylabel(r'% speedup', fontsize=13)


    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xticks(dims_to_xtick)
    ax.set_xticklabels(dims_to_xtick)
    ax.set_xticks([], minor=True)

    ax.set_title('  Linear layer summary, varying dimensions', fontsize=10, loc='left', y=1.05, pad=-20)



    plt.savefig('./plot_with_info_comparison.pdf', bbox_inches='tight')

