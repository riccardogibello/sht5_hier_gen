import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import torch

import plotly.graph_objects as go
import plotly.io as pio

# Force plotly to use browser
pio.renderers.default = "browser"


def average_attention_heads(
    attention_tensor: torch.Tensor,
) -> torch.Tensor:
    return attention_tensor.mean(dim=1)[:, -1:, :]


def plot_attention_matrix(
    matrix: torch.Tensor, x_labels, y_labels, title: str, save_path: str
):
    df = pd.DataFrame(matrix.cpu().numpy(), index=y_labels, columns=x_labels)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Attention Weight"},
    )
    plt.yticks(rotation=0)
    plt.title(title)
    plt.xlabel("Attended Tokens")
    plt.ylabel("Generated Tokens")
    plt.tight_layout()
    fig.savefig(save_path)


def build_decoder_self_attention_matrix(
    output_tokens: torch.Tensor,
    all_decoder_self_attns: torch.Tensor,
    output_folder_path: str,
):
    num_layers, max_len = len(all_decoder_self_attns[0]), len(output_tokens)
    matrix = torch.zeros((num_layers, max_len, max_len))

    for i, token_attn in enumerate(all_decoder_self_attns):
        for l, attn in enumerate(token_attn):
            avg_attn = average_attention_heads(attn)
            matrix[l, i, : avg_attn.shape[-1]] = avg_attn[0, 0, :]

    for l in range(num_layers):
        plot_attention_matrix(
            matrix[l],
            x_labels=output_tokens,
            y_labels=output_tokens,
            title=f"Decoder Self-Attention Matrix - Layer {l + 1}",
            save_path=os.path.join(
                output_folder_path, f"decoder_self_attention_matrix_layer_{l + 1}.png"
            ),
        )

    mean_matrix = matrix.mean(dim=0)
    plot_attention_matrix(
        mean_matrix,
        x_labels=output_tokens,
        y_labels=output_tokens,
        title="Mean Decoder Self-Attention Matrix",
        save_path=os.path.join(
            output_folder_path, "decoder_self_attention_matrix_mean.png"
        ),
    )

    return matrix


def collapse_tokens(tokens: list[str]) -> tuple[dict[int, list[int]], dict[int, str]]:
    new_idx_map, index_token, accumulated, idx = {}, {}, "", 0
    for orig_idx, tok in enumerate(tokens):
        if tok == "</s>":
            break
        if tok.startswith("▁"):
            accumulated = ""
            idx = max(new_idx_map.keys(), default=-1) + 1
        new_idx_map.setdefault(idx, []).append(orig_idx)
        next_tok = tokens[orig_idx + 1] if orig_idx + 1 < len(tokens) else None
        if next_tok is None or next_tok.startswith("▁") or next_tok == "</s>":
            index_token[idx] = accumulated + tok
        else:
            accumulated += tok
    return new_idx_map, index_token


def build_decoder_cross_attention_matrix(
    output_tokens: list[str],
    input_tokens: list[str],
    all_decoder_cross_attns: torch.Tensor,
    output_folder_path: str,
):
    output_tokens = [tok for tok in output_tokens if tok != "</s>"]
    new_idx_map, idx_to_tok = collapse_tokens(input_tokens)
    num_layers, out_len, in_len = (
        len(all_decoder_cross_attns[0]),
        len(output_tokens),
        len(new_idx_map),
    )
    matrix = torch.zeros((num_layers, out_len, in_len))

    for i, token_attn in enumerate(all_decoder_cross_attns):
        if i >= out_len:
            break
        for l, attn in enumerate(token_attn):
            avg_attn = average_attention_heads(attn)
            for new_j, old_indices in new_idx_map.items():
                matrix[l, i, new_j] = avg_attn[0, 0, old_indices].mean().item()

    for l in range(num_layers):
        plot_attention_matrix(
            matrix[l],
            x_labels=list(idx_to_tok.values()),
            y_labels=output_tokens,
            title=f"Decoder Cross-Attention Matrix - Layer {l + 1}",
            save_path=os.path.join(
                output_folder_path, f"decoder_cross_attention_matrix_layer_{l + 1}.png"
            ),
        )

    mean_matrix = matrix.mean(dim=0).T
    plot_attention_matrix(
        mean_matrix,
        x_labels=output_tokens,
        y_labels=list(idx_to_tok.values()),
        title="Mean Decoder Cross-Attention Matrix",
        save_path=os.path.join(
            output_folder_path, "decoder_cross_attention_matrix_mean.png"
        ),
    )

    return mean_matrix, output_tokens, list(idx_to_tok.values())


import torch
import plotly.graph_objects as go
import numpy as np
from matplotlib import cm


def attention_to_rgb_colors(attention_scores: list[float], cmap_name="viridis"):
    normed = (np.array(attention_scores) - np.min(attention_scores)) / (
        np.max(attention_scores) - np.min(attention_scores) + 1e-8
    )
    cmap = cm.get_cmap(cmap_name)
    return [
        f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"
        for r, g, b, a in cmap(normed)
    ]


def plot_interactive_attention_heatmap(
    cross_attention_matrix: torch.Tensor,
    output_tokens: list[str],
    merged_input_tokens: list[str],
    wrap_tokens_per_line: int = 15,
):
    cross_attention_matrix = cross_attention_matrix.cpu().numpy()
    num_output_tokens = len(output_tokens)
    num_input_tokens = len(merged_input_tokens)

    # Calculate token (x, y) positions for wrapping
    xs = [i % wrap_tokens_per_line for i in range(num_input_tokens)]
    ys = [-(i // wrap_tokens_per_line) for i in range(num_input_tokens)]

    # Prepare annotation sets for each output token
    annotations_sets = []
    for i in range(num_output_tokens):
        attn_scores = cross_attention_matrix[:, i]
        token_colors = attention_to_rgb_colors(attn_scores)

        annotations = []
        for j, token in enumerate(merged_input_tokens):
            annotations.append(
                dict(
                    x=xs[j],
                    y=ys[j] - 2,
                    xref="x",
                    yref="y",
                    text=token,
                    showarrow=False,
                    font=dict(
                        color="black",
                        size=16,
                        family="Montserrat, Arial, sans-serif",
                    ),
                    bgcolor=token_colors[j],
                    opacity=0.9,
                    align="center",
                )
            )
        annotations_sets.append(annotations)

    # Buttons to toggle annotations and update title annotation
    buttons = [
        dict(
            label=output_tokens[i],
            method="relayout",
            args=[
                {
                    "annotations": [
                        # Title annotation for selected token
                        dict(
                            text=f"<b>Attention to Input Tokens from Output Token: '{output_tokens[i]}'</b>",
                            x=0.25,
                            y=1.25,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16, family="Montserrat, Arial, sans-serif"),
                            align="left",
                        ),
                        # Under title: output tokens list (static, same for all states)
                        dict(
                            text="Output tokens: " + " ".join(output_tokens),
                            x=0.25,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=14, family="Montserrat, Arial, sans-serif"),
                            align="left",
                        ),
                        *annotations_sets[i],
                    ]
                }
            ],
        )
        for i in range(num_output_tokens)
    ]

    # Create empty figure
    fig = go.Figure()

    # Dummy invisible scatter trace for plot sizing and axes
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Layout with dropdown and annotations
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                showactive=True,
            )
        ],
        annotations=[
            dict(
                text=f"<b>Attention to Input Tokens from Output Token: '{output_tokens[0]}'</b>",
                x=0.25,
                y=1.25,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, family="Montserrat, Arial, sans-serif"),
                align="left",
            ),
            dict(
                text="Output tokens: " + " ".join(output_tokens),
                x=0.25,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, family="Montserrat, Arial, sans-serif"),
                align="left",
            ),
            *annotations_sets[0],
        ],
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.5, wrap_tokens_per_line - 0.5],
            fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[min(ys) - 5, 0.5],
            fixedrange=True,
        ),
        height=max(350, (abs(min(ys)) + 1) * 40),
        margin=dict(l=20, r=20, t=140, b=20),  # extra top margin for output tokens text
        plot_bgcolor="white",
    )

    fig.show()
