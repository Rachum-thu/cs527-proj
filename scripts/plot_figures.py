"""Generate paper-quality figures for the fidelity experiment."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Use a clean, publication-ready style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
})

OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)

# ============================================================
# Figure: CER vs Token Cost (font size sweep) — dual y-axis
# ============================================================

font_sizes = [8, 10, 12, 14, 16, 18, 20]
labels = ['f8', 'f10', 'f12', 'f14', 'f16', 'f18', 'f20']
cer = [0.084, 0.046, 0.023, 0.019, 0.030, 0.018, 0.027]
tokens = [538, 700, 1038, 1338, 1853, 2161, 2349]

fig, ax1 = plt.subplots(figsize=(3.4, 2.4))

color_cer = '#c0392b'
color_tok = '#2c3e6b'

# CER line
line1 = ax1.plot(font_sizes, cer, 'o-', color=color_cer, label='Avg CER',
                 markerfacecolor='white', markeredgewidth=1.2, zorder=3)
ax1.set_xlabel('Font Size (pt)')
ax1.set_ylabel('Character Error Rate (CER)', color=color_cer)
ax1.tick_params(axis='y', labelcolor=color_cer)
ax1.set_ylim(0, 0.10)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

# Highlight sweet spot
ax1.axvspan(11.5, 14.5, alpha=0.08, color='#27ae60', zorder=0)
ax1.annotate('sweet\nspot', xy=(13, 0.005), fontsize=7, color='#1a7a4c',
             ha='center', style='italic')

# Token line on secondary axis
ax2 = ax1.twinx()
line2 = ax2.plot(font_sizes, tokens, 's--', color=color_tok, label='Avg Tokens',
                 markerfacecolor='white', markeredgewidth=1.2, markersize=4, zorder=2)
ax2.set_ylabel('Avg Input Tokens', color=color_tok)
ax2.tick_params(axis='y', labelcolor=color_tok)
ax2.set_ylim(0, 2800)

# Combined legend
lines = line1 + line2
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc='upper center', framealpha=0.9, edgecolor='#ddd')

ax1.set_xticks(font_sizes)
ax1.set_xticklabels(labels)
ax1.grid(axis='y', alpha=0.2, linewidth=0.4)

plt.tight_layout()
fig.savefig(f'{OUT}/cer_vs_fontsize.pdf')
fig.savefig(f'{OUT}/cer_vs_fontsize.png')
print(f"Saved cer_vs_fontsize")

# ============================================================
# Figure: Page Width comparison (grouped bar)
# ============================================================

settings = ['f12\nw100', 'f12\nw120', 'f14\nw100', 'f14\nw120', 'f14\nw140']
cer_vals = [0.023, 0.023, 0.019, 0.018, 0.015]
indent_vals = [0.99, 1.00, 0.82, 0.83, 0.83]
tok_vals = [1038, 1160, 1338, 1524, 1762]

x = np.arange(len(settings))
width = 0.35

fig, ax1 = plt.subplots(figsize=(3.4, 2.2))

bars1 = ax1.bar(x - width/2, cer_vals, width, label='CER ↓', color=color_cer, alpha=0.85, edgecolor='white')
ax1.set_ylabel('CER', color=color_cer)
ax1.tick_params(axis='y', labelcolor=color_cer)
ax1.set_ylim(0, 0.035)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, indent_vals, width, label='Indent Acc ↑', color='#27ae60', alpha=0.85, edgecolor='white')
ax2.set_ylabel('Indent Accuracy', color='#1a7a4c')
ax2.tick_params(axis='y', labelcolor='#1a7a4c')
ax2.set_ylim(0.7, 1.05)

ax1.set_xticks(x)
ax1.set_xticklabels(settings, fontsize=7.5)
ax1.set_xlabel('Configuration')

# Highlight best
bars1[1].set_edgecolor('#c0392b')
bars1[1].set_linewidth(1.5)
bars2[1].set_edgecolor('#1a7a4c')
bars2[1].set_linewidth(1.5)

lines = [bars1, bars2]
labs = ['CER ↓', 'Indent Acc ↑']
ax1.legend(lines, labs, loc='upper left', framealpha=0.9, edgecolor='#ddd', fontsize=7)

ax1.grid(axis='y', alpha=0.15, linewidth=0.4)
plt.tight_layout()
fig.savefig(f'{OUT}/pagewidth_comparison.pdf')
fig.savefig(f'{OUT}/pagewidth_comparison.png')
print(f"Saved pagewidth_comparison")

# ============================================================
# Figure: Per-sample CER for f12_w120 (horizontal bar)
# ============================================================

samples = ['dense_imports\n(46L)', 'agent_short\n(50L)', 'special_chars\n(58L)',
           'complex_logic\n(96L)', 'agent_medium\n(100L)', 'agent_full\n(155L)']
sample_cer = [0.001, 0.015, 0.094, 0.011, 0.010, 0.009]
colors = ['#27ae60' if c < 0.02 else '#d4a017' if c < 0.05 else '#c0392b' for c in sample_cer]

fig, ax = plt.subplots(figsize=(3.4, 2.0))
y = np.arange(len(samples))
bars = ax.barh(y, sample_cer, height=0.6, color=colors, alpha=0.85, edgecolor='white')
ax.set_yticks(y)
ax.set_yticklabels(samples, fontsize=7)
ax.set_xlabel('Character Error Rate (CER)')
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, 0.12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.2, linewidth=0.4)

# Annotate values
for bar, val in zip(bars, sample_cer):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=7, color='#444')

ax.set_title('Per-Sample CER — f12, w120 config', fontsize=9, fontweight='bold', pad=8)
plt.tight_layout()
fig.savefig(f'{OUT}/per_sample_cer.pdf')
fig.savefig(f'{OUT}/per_sample_cer.png')
print(f"Saved per_sample_cer")

plt.close('all')
print("\nAll figures saved to figures/")
