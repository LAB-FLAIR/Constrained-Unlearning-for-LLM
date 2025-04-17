import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def plot(dir):
    np_files = os.listdir(dir)
    heights = {}
    mapper = {
        'initial_tokens': [],
                'sub_1': [],
            'sub_2': [],
            'sub_3': [],
            'last_1': [],
            'last_2': [],
            'last_3': []}
    for file in np_files:
        ops = np.load(os.path.join(dir, file))
        subj_range = ops['subject_range']

        initial_tokens = ops['scores'][:subj_range[0]].mean(axis=0)

        avg_subject_token_1 = ops['scores'][subj_range[0]]
        avg_subject_token_mid = ops['scores'][subj_range[0]+1:subj_range[1]].mean(axis=0)
        avg_subject_token_3 = ops['scores'][subj_range[0]]

        first_subseq_token = ops['scores'][subj_range[1]+1]
        mid_subseq_token = ops['scores'][subj_range[1]+1:len(ops['input_ids'])-1].mean(axis=0)
        last_subseq_token = ops['scores'][len(ops['input_ids'])-1]

        # height = ops["scores"][subj_range[0]:subj_range[1]].mean(axis=0)
        kind = ops['kind'].item()
        if kind in heights.keys():
            heights[kind]['initial_tokens'].append(initial_tokens)
            heights[kind]['sub_1'].append(avg_subject_token_1)
            heights[kind]['sub_2'].append(avg_subject_token_mid)
            heights[kind]['sub_3'].append(avg_subject_token_3)
            heights[kind]['last_1'].append(first_subseq_token)
            heights[kind]['last_2'].append(mid_subseq_token)
            heights[kind]['last_3'].append(last_subseq_token)
        else:
            heights[kind] = deepcopy(mapper)
            heights[kind]['initial_tokens'].append(initial_tokens)
            heights[kind]['sub_1'].append(avg_subject_token_1)
            heights[kind]['sub_2'].append(avg_subject_token_mid)
            heights[kind]['sub_3'].append(avg_subject_token_3)
            heights[kind]['last_1'].append(first_subseq_token)
            heights[kind]['last_2'].append(mid_subseq_token)
            heights[kind]['last_3'].append(last_subseq_token)

    for k,v in heights.items():
        temp = []
        for k_sub, v_sub in heights[k].items():
            temp.append(np.array(v_sub).mean(axis=0))
        heights[k] = np.array(temp)

    heights[None] = heights['']
    kind_mapper = {
        'self_attn': 'Attention',
        '': None,
        'mlp': 'MLP'
    }


    modelname = 'OLMo'
    window = 10

    with plt.rc_context():
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=200)

        # Plot for 'self_attn'
        kind = 'self_attn'
        h1 = axes[0].pcolor(
            heights[kind][::-1],
            cmap="Reds",
            # vmin=0, vmax=1
        )
        axes[0].set_yticks([0.5 + i for i in range(len(heights[kind]))])
        axes[0].set_xticks([0.5 + i for i in range(0, heights[kind].shape[1] - 6, 5)])
        axes[0].set_xticklabels(list(range(0, heights[kind].shape[1] - 6, 5)))
        # Reverse the y-axis labels to match data order
        axes[0].set_yticklabels([
            r'$r_l$', r'$r_m$', r'$r_f$',
            r'$s_l$', r'$s_m$', r'$s_f$',
            r'$i$'
        ])
        axes[0].set_title("Impact of restoring MHSA")

        # Plot for 'mlp'
        kind = 'mlp'
        h2 = axes[1].pcolor(
            heights[kind][::-1],
            cmap="Greens",
            # vmin=0, vmax=1
        )
        axes[1].set_xticks([0.5 + i for i in range(0, heights[kind].shape[1] - 6, 5)])
        axes[1].set_xticklabels(list(range(0, heights[kind].shape[1] - 6, 5)))
        axes[1].set_yticks([0.5 + i for i in range(len(heights[kind]))])
        axes[1].set_yticklabels([
            r'$r_l$', r'$r_m$', r'$r_f$',
            r'$s_l$', r'$s_m$', r'$s_f$',
            r'$i$'
        ])

        axes[1].set_yticklabels([
            r'$r_l$', r'$r_m$', r'$r_f$',
            r'$s_l$', r'$s_m$', r'$s_f$',
            r'$i$'
        ])
        axes[1].set_title("Impact of restoring MLP")
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        num_rows, num_cols = heights[kind][::-1].shape
        # Add colorbars
        fig.colorbar(h1, ax=axes[0], pad=0.04)
        fig.colorbar(h2, ax=axes[1], pad=0.04)
        
        fig.savefig("causal_graph.png", dpi=150)
