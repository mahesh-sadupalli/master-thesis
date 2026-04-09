"""
Generate high-quality architecture diagrams for mid-term presentation.
Replicating mentor's visual style: 3D cubes, vertical layer bars, data tables,
curly brackets, loss box, consideration callout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import numpy as np
import os

OUT = '/Users/mahesh/Desktop/master-thesis/presentations/diagrams'
os.makedirs(OUT, exist_ok=True)

# ─── Mentor's color palette ───
GREEN_FRONT = '#2E8B57'
GREEN_TOP   = '#3CB371'
GREEN_SIDE  = '#228B22'
BLUE_FRONT  = '#006EB8'
BLUE_TOP    = '#4DA6E8'
BLUE_SIDE   = '#004C8C'
ORANGE_FRONT = '#E88D2A'
ORANGE_TOP   = '#FFB347'
ORANGE_SIDE  = '#CC7722'
PURPLE = '#4B0082'
DARK = '#333333'
RED_TEXT = '#DC3545'
LOSS_BG = '#FFF8E1'
LOSS_BORDER = '#E88D2A'
CALLOUT_BG = '#FDEBD0'
CALLOUT_BORDER = '#E88D2A'
LAYER_GREEN = '#2E8B57'
LAYER_PURPLE = '#6A0DAD'
LATENT_BG = '#8B0000'
BG = '#F0F0F0'


def draw_cube(ax, x, y, w, h, d=0.35, fc='#2E8B57', tc='#3CB371', sc='#228B22',
              label='', label_fs=11, lc='white'):
    """3D cube matching mentor style."""
    dx, dy = d * w, d * h
    # Front
    ax.add_patch(plt.Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                             fc=fc, ec='#222', lw=1.5, zorder=3))
    # Top with subtle lines
    ax.add_patch(plt.Polygon([(x, y+h), (x+w, y+h), (x+w+dx, y+h+dy), (x+dx, y+h+dy)],
                             fc=tc, ec='#222', lw=1.5, zorder=3))
    for i in range(1, 5):
        xi = x + i * w / 5
        ax.plot([xi, xi+dx], [y+h, y+h+dy], color='white', lw=0.4, alpha=0.4, zorder=4)
    # Right side
    ax.add_patch(plt.Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                             fc=sc, ec='#222', lw=1.5, zorder=3))
    if label:
        lines = label.split('\n')
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h/2 + (len(lines)/2 - i - 0.5) * 0.35,
                    line, ha='center', va='center', fontsize=label_fs,
                    fontweight='bold', color=lc, zorder=5)


def draw_layer_bar(ax, x, y_bottom, width, height, color=LAYER_GREEN):
    """Single vertical bar with gradient highlight."""
    bar = FancyBboxPatch((x, y_bottom), width, height,
                         boxstyle="round,pad=0.01", fc=color, ec='#1A5030',
                         lw=0.8, zorder=3)
    ax.add_patch(bar)
    # Highlight stripe
    hl = FancyBboxPatch((x + 0.02, y_bottom + 0.05), width * 0.25, height - 0.1,
                        boxstyle="round,pad=0.005", fc='white', alpha=0.15,
                        ec='none', zorder=4)
    ax.add_patch(hl)


def draw_layer_group(ax, x_center, y_center, n_bars, bar_h, bar_w=0.1,
                     gap=0.05, color=LAYER_GREEN):
    """Group of vertical bars centered at (x_center, y_center)."""
    total_w = n_bars * bar_w + (n_bars - 1) * gap
    x_start = x_center - total_w / 2
    y_bottom = y_center - bar_h / 2
    for i in range(n_bars):
        bx = x_start + i * (bar_w + gap)
        draw_layer_bar(ax, bx, y_bottom, bar_w, bar_h, color)
    return x_start, x_start + total_w  # return left and right edges


def draw_thick_arrow(ax, x1, y, x2, color='#555', lw=3):
    """Thick horizontal arrow."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=25))


def draw_bracket_above(ax, x1, x2, y_top, label, label_fs=9.5):
    """Curly bracket above a group with label."""
    mid = (x1 + x2) / 2
    bh = 0.2
    ax.plot([x1, x1], [y_top, y_top + bh], color=DARK, lw=1.2, zorder=5)
    ax.plot([x2, x2], [y_top, y_top + bh], color=DARK, lw=1.2, zorder=5)
    ax.plot([x1, mid - 0.1], [y_top + bh, y_top + bh], color=DARK, lw=1.2, zorder=5)
    ax.plot([mid + 0.1, x2], [y_top + bh, y_top + bh], color=DARK, lw=1.2, zorder=5)
    ax.plot([mid - 0.1, mid], [y_top + bh, y_top + bh + 0.1], color=DARK, lw=1.2, zorder=5)
    ax.plot([mid + 0.1, mid], [y_top + bh, y_top + bh + 0.1], color=DARK, lw=1.2, zorder=5)
    ax.text(mid, y_top + bh + 0.2, label, ha='center', va='bottom',
            fontsize=label_fs, fontweight='bold', color=DARK, zorder=5)


def draw_bracket_below(ax, x1, x2, y_bot, label, label_fs=9):
    """Bracket below a group with label."""
    mid = (x1 + x2) / 2
    bh = 0.15
    ax.plot([x1, x1], [y_bot, y_bot - bh], color=DARK, lw=1, zorder=5)
    ax.plot([x2, x2], [y_bot, y_bot - bh], color=DARK, lw=1, zorder=5)
    ax.plot([x1, x2], [y_bot - bh, y_bot - bh], color=DARK, lw=1, zorder=5)
    ax.text(mid, y_bot - bh - 0.12, label, ha='center', va='top',
            fontsize=label_fs, fontweight='bold', color=DARK, zorder=5)


def draw_data_table(ax, x, y, headers, row1, row2, header_bg=PURPLE,
                    highlight_cols=None, cw=0.55, ch=0.28):
    """Data table like mentor's diagram."""
    nc = len(headers)
    # Header
    for i, h in enumerate(headers):
        cx = x + i * cw
        ax.add_patch(FancyBboxPatch((cx, y + ch), cw, ch, boxstyle="square,pad=0",
                                    fc=header_bg, ec='white', lw=0.5, zorder=3))
        ax.text(cx + cw/2, y + ch * 1.5, h, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)
    # Row 1
    for i, v in enumerate(row1):
        cx = x + i * cw
        ax.add_patch(FancyBboxPatch((cx, y), cw, ch, boxstyle="square,pad=0",
                                    fc='white', ec='#CCC', lw=0.5, zorder=3))
        tc = RED_TEXT if (highlight_cols and i in highlight_cols) else DARK
        fw = 'bold' if (highlight_cols and i in highlight_cols) else 'normal'
        ax.text(cx + cw/2, y + ch/2, str(v), ha='center', va='center',
                fontsize=7.5, color=tc, fontweight=fw, zorder=4)
    # Row 2
    for i, v in enumerate(row2):
        cx = x + i * cw
        ax.add_patch(FancyBboxPatch((cx, y - ch), cw, ch, boxstyle="square,pad=0",
                                    fc='#F5F5F5', ec='#CCC', lw=0.5, zorder=3))
        ax.text(cx + cw/2, y - ch/2, str(v), ha='center', va='center',
                fontsize=7.5, color='#888', zorder=4)


# ═══════════════════════════════════════════════════════
# DIAGRAM 1: INR Architecture (replicating mentor style)
# ═══════════════════════════════════════════════════════
def create_inr_architecture():
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(-1, 18)
    ax.set_ylim(-2.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    cy = 2.5  # center y for layers

    # ── LEFT: Input Data Cube ──
    draw_cube(ax, 0, 1.2, 2.0, 2.6, d=0.3,
              fc=BLUE_FRONT, tc=BLUE_TOP, sc=BLUE_SIDE,
              label='Input\nData', label_fs=13)

    # Input table below
    draw_data_table(ax, -0.3, -1.2,
                    ['X', 'Y', 'Z', 'Time'],
                    ['1.01', '2.01', '5.02', '0.01'],
                    ['N...', 'N...', 'N...', 'N...'],
                    header_bg=BLUE_FRONT)

    # ── ARROW: Input → Layers ──
    draw_thick_arrow(ax, 2.6, cy, 3.8)

    # ── NETWORK LAYERS (4 → 64 → 64 → 32 → 4) ──
    # Layer 1: Input (4 neurons) - small
    l1_left, l1_right = draw_layer_group(ax, 4.5, cy, 3, 1.2, 0.1, 0.05, LAYER_PURPLE)

    # Layer 2: Hidden (64 neurons) - tall
    l2_left, l2_right = draw_layer_group(ax, 6.2, cy, 6, 3.0, 0.1, 0.05, LAYER_GREEN)

    # Layer 3: Hidden (64 neurons) - tall
    l3_left, l3_right = draw_layer_group(ax, 8.2, cy, 6, 3.0, 0.1, 0.05, LAYER_GREEN)

    # Layer 4: Hidden (32 neurons) - medium
    l4_left, l4_right = draw_layer_group(ax, 10.2, cy, 5, 2.0, 0.1, 0.05, LAYER_GREEN)

    # Layer 5: Output (4 neurons) - small
    l5_left, l5_right = draw_layer_group(ax, 12.0, cy, 3, 1.2, 0.1, 0.05, LAYER_PURPLE)

    # Arrows between layer groups
    for x1, x2 in [(l1_right, l2_left), (l2_right, l3_left),
                    (l3_right, l4_left), (l4_right, l5_left)]:
        draw_thick_arrow(ax, x1 + 0.05, cy, x2 - 0.05, color='#AAA', lw=1.5)

    # Layer size labels below
    for xc, label in [(4.5, '4'), (6.2, '64'), (8.2, '64'), (10.2, '32'), (12.0, '4')]:
        ax.text(xc, cy - 1.8, label, ha='center', va='top',
                fontsize=11, fontweight='bold', color=DARK)

    # ── BRACKET: Hidden Layers ──
    draw_bracket_above(ax, l2_left - 0.05, l4_right + 0.05, cy + 1.7,
                       'Hidden Layers: ReLU Activation', label_fs=11)

    # ── Bracket below layer groups ──
    draw_bracket_below(ax, l1_left, l1_right, cy - 1.2, 'Input\nLayer', label_fs=8)
    draw_bracket_below(ax, l2_left, l3_right, cy - 1.2, 'Hidden Layers\n(ReLU)', label_fs=8)
    draw_bracket_below(ax, l4_left, l4_right, cy - 1.2, 'Hidden\nLayer', label_fs=8)
    draw_bracket_below(ax, l5_left, l5_right, cy - 1.2, 'Output\nLayer', label_fs=8)

    # ── ARROW: Layers → Output ──
    draw_thick_arrow(ax, l5_right + 0.2, cy, 13.5)

    # ── RIGHT: Predicted Data Cube ──
    draw_cube(ax, 13.7, 1.2, 2.0, 2.6, d=0.3,
              fc=ORANGE_FRONT, tc=ORANGE_TOP, sc=ORANGE_SIDE,
              label='Predicted\nData', label_fs=13)

    # Output table below (red for predicted values)
    draw_data_table(ax, 13.3, -1.2,
                    ['V\u0302x', 'V\u0302y', 'P\u0302', 'TKE\u0302'],
                    ['1.05', '-0.32', '0.95', '12.3'],
                    ['N...', 'N...', 'N...', 'N...'],
                    header_bg=ORANGE_FRONT, highlight_cols=[0, 1, 2, 3])

    # ── LOSS BOX (center bottom) ──
    loss_box = FancyBboxPatch((6.5, -1.6), 4.0, 0.9,
                              boxstyle="round,pad=0.15", fc=LOSS_BG,
                              ec=LOSS_BORDER, lw=2, zorder=3)
    ax.add_patch(loss_box)
    ax.text(8.5, -1.0, 'Loss: MSE minimization', ha='center', va='center',
            fontsize=10, fontweight='bold', color=DARK, zorder=4)
    ax.text(8.5, -1.35, 'Reconstruction: PSNR, SSIM, Rel. Error', ha='center', va='center',
            fontsize=9, color='#666', zorder=4)

    # ── CALLOUT BOX (bottom) ──
    callout = FancyBboxPatch((3.5, -2.4), 10.0, 0.55,
                             boxstyle="round,pad=0.1", fc=CALLOUT_BG,
                             ec=CALLOUT_BORDER, lw=1.5, zorder=3)
    ax.add_patch(callout)
    ax.text(8.5, -2.12, 'Architecture: 4 \u2192 64 \u2192 64 \u2192 32 \u2192 4  |  '
            'Params: 6,692  |  Size: 26.1 KB  |  Compression: ~4,734 : 1',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='#8B4000', zorder=4)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, 'arch_inr.png'), dpi=220, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('Created: arch_inr.png')


# ═══════════════════════════════════════════════════════
# DIAGRAM 2: Three Model Sizes (mentor bar style)
# ═══════════════════════════════════════════════════════
def create_model_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(17, 9.5))
    fig.patch.set_facecolor(BG)

    models = [
        ('BaseCompressor', [4, 64, 64, 32, 4], [3, 6, 6, 5, 3],
         [1.0, 2.8, 2.8, 1.8, 1.0], '6,692', '26.1 KB', '~4,734:1', BLUE_FRONT),
        ('MediumCompressor', [4, 96, 96, 48, 4], [3, 7, 7, 5, 3],
         [1.0, 3.2, 3.2, 2.2, 1.0], '14,644', '57.2 KB', '~2,165:1', GREEN_FRONT),
        ('LargeCompressor', [4, 128, 128, 64, 4], [3, 8, 8, 6, 3],
         [1.0, 3.5, 3.5, 2.5, 1.0], '25,668', '100.3 KB', '~1,233:1', ORANGE_FRONT),
    ]

    for idx, (name, sizes, n_bars_list, heights, params, sz, cr, accent) in enumerate(models):
        ax = axes[idx]
        ax.set_xlim(0, 17)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')

        cy = 1.5

        # Model badge
        badge = FancyBboxPatch((0.2, 0.7), 2.5, 1.5,
                               boxstyle="round,pad=0.15", fc=accent, ec='#222',
                               lw=1.5, zorder=3)
        ax.add_patch(badge)
        ax.text(1.45, 1.45, name, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', zorder=4)

        # Arrow from badge
        draw_thick_arrow(ax, 2.8, cy, 3.5, color='#AAA', lw=2)

        # Draw layers
        x_positions = [4.2, 5.8, 7.4, 9.0, 10.6]
        layer_colors = [LAYER_PURPLE, LAYER_GREEN, LAYER_GREEN, LAYER_GREEN, LAYER_PURPLE]

        for i, (xc, nb, bh, lc) in enumerate(zip(x_positions, n_bars_list, heights, layer_colors)):
            draw_layer_group(ax, xc, cy, nb, bh, 0.1, 0.05, lc)
            # Size label
            ax.text(xc, cy - max(heights) / 2 - 0.35, str(sizes[i]),
                    ha='center', fontsize=10, fontweight='bold', color=DARK)
            # Arrow
            if i < 4:
                draw_thick_arrow(ax, xc + 0.6, cy, x_positions[i+1] - 0.6,
                                 color='#CCC', lw=1.2)

        # ReLU labels
        for i in range(3):
            mid_x = (x_positions[i] + x_positions[i+1]) / 2
            ax.text(mid_x, cy + max(heights) / 2 + 0.15, 'ReLU',
                    ha='center', fontsize=8, fontweight='bold', color='#FF6934')

        # Stats on right
        sx = 12.2
        ax.text(sx, 2.2, params + ' parameters', fontsize=12, fontweight='bold', color=DARK)
        ax.text(sx, 1.6, 'Model Size: ' + sz, fontsize=11, color='#666')
        ax.text(sx, 1.0, 'Compression Ratio: ' + cr, fontsize=11,
                fontweight='bold', color=accent)

        # Architecture string
        arch = ' \u2192 '.join(str(s) for s in sizes)
        ax.text(7.4, -0.3, arch, ha='center', fontsize=9, color='#999', fontstyle='italic')

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, 'arch_models.png'), dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('Created: arch_models.png')


# ═══════════════════════════════════════════════════════
# DIAGRAM 3: Online Temporal Windows
# ═══════════════════════════════════════════════════════
def create_online_diagram():
    fig, ax = plt.subplots(figsize=(17, 6.5))
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(-2, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    cy = 2.0
    win_w, win_h = 1.6, 2.5

    # Title
    ax.text(8.25, 5.0, 'Sequential Temporal Window Training',
            ha='center', fontsize=16, fontweight='bold', color=DARK)
    ax.text(8.25, 4.5, '20 windows  \u00d7  100 epochs/window  =  2,000 total epochs',
            ha='center', fontsize=11, color='#666')

    # Windows
    blues = ['#B8D4E8', '#8BB8D9', '#5E9CCA', '#3180BB', '#006EB8']
    labels_ts = ['t1-t15', 't16-t30', 't31-t45', 't46-t60', 't61-t75']

    for i in range(5):
        x = 0.3 + i * (win_w + 0.25)
        alpha = 0.4 + i * 0.1
        fc = blues[i]
        rect = FancyBboxPatch((x, cy - win_h/2), win_w, win_h,
                              boxstyle="round,pad=0.08", fc=fc, ec='#3366AA',
                              lw=1.5, alpha=alpha, zorder=3)
        ax.add_patch(rect)
        ax.text(x + win_w/2, cy + 0.5, 'W{}'.format(i+1), ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', alpha=min(1, alpha+0.3), zorder=4)
        ax.text(x + win_w/2, cy, labels_ts[i], ha='center', va='center',
                fontsize=8, color='white', alpha=min(1, alpha+0.2), zorder=4)
        ax.text(x + win_w/2, cy - 0.5, '100 epochs', ha='center', va='center',
                fontsize=7, color='white', alpha=min(1, alpha+0.1), zorder=4)

        # Forgotten label
        ax.text(x + win_w/2, cy - win_h/2 - 0.25, '\u2717 Forgotten',
                ha='center', fontsize=8, fontweight='bold', color=RED_TEXT)

        # Arrow to next
        if i < 4:
            ax.annotate('', xy=(x + win_w + 0.2, cy), xytext=(x + win_w + 0.05, cy),
                        arrowprops=dict(arrowstyle='->', color='#4A6FA5', lw=2))

    # Dots
    dots_x = 0.3 + 5 * (win_w + 0.25)
    ax.text(dots_x + 0.3, cy, '\u2022 \u2022 \u2022', ha='center', va='center',
            fontsize=20, color='#888', fontweight='bold')

    # Arrow from dots to W20
    ax.annotate('', xy=(dots_x + 1.2, cy), xytext=(dots_x + 0.7, cy),
                arrowprops=dict(arrowstyle='->', color='#4A6FA5', lw=2))

    # W20 (green, active)
    w20_x = dots_x + 1.4
    rect20 = FancyBboxPatch((w20_x, cy - win_h/2), win_w, win_h,
                            boxstyle="round,pad=0.08", fc='#2E8B57', ec='#1A5030',
                            lw=2.5, zorder=3)
    ax.add_patch(rect20)
    ax.text(w20_x + win_w/2, cy + 0.5, 'W20', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white', zorder=4)
    ax.text(w20_x + win_w/2, cy, 't286-t300', ha='center', va='center',
            fontsize=9, color='white', zorder=4)
    ax.text(w20_x + win_w/2, cy - 0.5, '100 epochs', ha='center', va='center',
            fontsize=8, color='white', zorder=4)
    ax.text(w20_x + win_w/2, cy - win_h/2 - 0.25, '\u2713 Learned',
            ha='center', fontsize=8, fontweight='bold', color='#2E8B57')

    # Warning callout
    callout = FancyBboxPatch((2.5, -1.8), 12.0, 0.7,
                             boxstyle="round,pad=0.12", fc='#FDECEA',
                             ec=RED_TEXT, lw=2, zorder=3)
    ax.add_patch(callout)
    ax.text(8.5, -1.45, '\u26A0  Catastrophic Forgetting: Model optimized for Window 20 only '
            '\u2014 forgets Windows 1\u201319',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=RED_TEXT, zorder=4)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, 'arch_online.png'), dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('Created: arch_online.png')


# ═══════════════════════════════════════════════════════
# DIAGRAM 4: Offline vs Online side-by-side
# ═══════════════════════════════════════════════════════
def create_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.5))
    fig.patch.set_facecolor(BG)

    for ax, title, color, data_label, model_label, result, result_color, pros, cons in [
        (ax1, 'Offline (Batch) Training', BLUE_FRONT,
         'Full Dataset\n7.9M samples\nAll 300 timesteps', 'MLP f\u03B8\n150 epochs',
         '35.99 dB\nPSNR', GREEN_FRONT,
         '\u2713 Best quality    \u2713 Stable training',
         '\u2717 Requires all data upfront'),
        (ax2, 'Online (Streaming) Training', RED_TEXT,
         'Window k\n~396K samples\n15 timesteps', 'MLP f\u03B8\n100 epochs\n/window',
         '9.67 dB\nPSNR\n(full eval)', RED_TEXT,
         '\u2713 Streaming capable    \u2713 Low memory',
         '\u2717 Catastrophic forgetting (~20 dB loss)')
    ]:
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 5.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        ax.text(4, 5.2, title, ha='center', fontsize=15, fontweight='bold', color=color)

        # Data box
        data_box = FancyBboxPatch((0.3, 2.3), 2.3, 1.8,
                                  boxstyle="round,pad=0.12",
                                  fc=color if color == BLUE_FRONT else '#CC4444',
                                  ec='#333', lw=1.5, zorder=3)
        ax.add_patch(data_box)
        ax.text(1.45, 3.2, data_label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

        # Arrow
        ax.annotate('', xy=(3.3, 3.2), xytext=(2.7, 3.2),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.5))

        # Model box
        model_box = FancyBboxPatch((3.4, 2.5), 1.8, 1.4,
                                   boxstyle="round,pad=0.1", fc=GREEN_FRONT, ec='#1A5030',
                                   lw=1.5, zorder=3)
        ax.add_patch(model_box)
        ax.text(4.3, 3.2, model_label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

        # Arrow
        ax.annotate('', xy=(5.9, 3.2), xytext=(5.3, 3.2),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.5))

        # Result box
        res_box = FancyBboxPatch((6.0, 2.3), 1.7, 1.8,
                                 boxstyle="round,pad=0.12", fc=result_color, ec='#333',
                                 lw=1.5, zorder=3)
        ax.add_patch(res_box)
        ax.text(6.85, 3.2, result, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold', zorder=4)

        # Pros
        ax.text(4, 1.5, pros, ha='center', fontsize=10, fontweight='bold', color=GREEN_FRONT)
        ax.text(4, 0.9, cons, ha='center', fontsize=10, fontweight='bold', color=RED_TEXT)

    # Separator
    fig.patches.append(plt.Rectangle((0.498, 0.08), 0.004, 0.84,
                                      transform=fig.transFigure, fc='#CCC', ec='none'))

    plt.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUT, 'comparison.png'), dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('Created: comparison.png')


# ═══════════════════════════════════════════════════════
# DIAGRAM 5: Compression Concept
# ═══════════════════════════════════════════════════════
def create_compression():
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    # Big original cube
    draw_cube(ax, 0.3, 0.5, 2.8, 2.8, d=0.3,
              fc=BLUE_FRONT, tc=BLUE_TOP, sc=BLUE_SIDE,
              label='Original\nDataset\n241.6 MB', label_fs=12)

    # Arrow: Encode
    draw_thick_arrow(ax, 3.8, 1.9, 5.5, color=GREEN_FRONT, lw=3)
    ax.text(4.65, 2.5, 'Encode\n(Train)', ha='center', fontsize=10,
            fontweight='bold', color=GREEN_FRONT)

    # Small model box
    model = FancyBboxPatch((5.7, 1.0), 1.8, 1.8,
                           boxstyle="round,pad=0.15", fc=GREEN_FRONT, ec='#1A5030',
                           lw=2, zorder=3)
    ax.add_patch(model)
    ax.text(6.6, 1.9, 'Model \u03B8\n26.1 KB', ha='center', va='center',
            fontsize=12, color='white', fontweight='bold', zorder=4)

    # Arrow: Decode
    draw_thick_arrow(ax, 7.7, 1.9, 9.4, color=ORANGE_FRONT, lw=3)
    ax.text(8.55, 2.5, 'Decode\n(Inference)', ha='center', fontsize=10,
            fontweight='bold', color=ORANGE_FRONT)

    # Big reconstructed cube
    draw_cube(ax, 9.6, 0.5, 2.8, 2.8, d=0.3,
              fc=ORANGE_FRONT, tc=ORANGE_TOP, sc=ORANGE_SIDE,
              label='Reconstructed\nDataset\n~241.6 MB', label_fs=12)

    # Compression ratio callout
    cr_box = FancyBboxPatch((13.3, 0.6), 2.5, 2.5,
                            boxstyle="round,pad=0.2", fc='#FFF3CD',
                            ec=ORANGE_FRONT, lw=2.5, zorder=3)
    ax.add_patch(cr_box)
    ax.text(14.55, 2.5, 'Compression', ha='center', fontsize=11,
            fontweight='bold', color=DARK, zorder=4)
    ax.text(14.55, 1.8, '4,734 : 1', ha='center', fontsize=18,
            fontweight='bold', color=ORANGE_FRONT, zorder=4)
    ax.text(14.55, 1.15, '(BaseCompressor)', ha='center', fontsize=9,
            color='#888', zorder=4)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, 'compression.png'), dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('Created: compression.png')


# ═══════════════════════════════════════════════════════
# DIAGRAM 6: Compression Ratio Calculation
# ═══════════════════════════════════════════════════════
def create_compression_calc():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    # Title
    ax.text(8, 7.5, 'Compression Ratio Calculation', ha='center',
            fontsize=16, fontweight='bold', color=DARK)

    # Original data calc
    box1 = FancyBboxPatch((0.5, 4.8), 7, 2.2, boxstyle="round,pad=0.2",
                          fc='white', ec=BLUE_FRONT, lw=2, zorder=3)
    ax.add_patch(box1)
    ax.text(4, 6.5, 'Original Dataset Size', ha='center', fontsize=13,
            fontweight='bold', color=BLUE_FRONT, zorder=4)
    ax.text(4, 5.9, '7,919,100 rows  \u00d7  4 target columns  \u00d7  4 bytes (float32)',
            ha='center', fontsize=10, color=DARK, zorder=4)
    ax.text(4, 5.35, '= 126,705,600 bytes  \u2248  120.8 MB',
            ha='center', fontsize=12, fontweight='bold', color=BLUE_FRONT, zorder=4)

    # Model sizes
    box2 = FancyBboxPatch((8.5, 4.8), 7, 2.2, boxstyle="round,pad=0.2",
                          fc='white', ec=GREEN_FRONT, lw=2, zorder=3)
    ax.add_patch(box2)
    ax.text(12, 6.5, 'Model Size (float32)', ha='center', fontsize=13,
            fontweight='bold', color=GREEN_FRONT, zorder=4)
    ax.text(12, 5.9, 'Base:   6,692 params \u00d7 4 bytes  =  26,768 bytes  \u2248  26.1 KB',
            ha='center', fontsize=9, color=DARK, zorder=4)
    ax.text(12, 5.5, 'Medium: 14,644 params \u00d7 4 bytes  =  58,576 bytes  \u2248  57.2 KB',
            ha='center', fontsize=9, color=DARK, zorder=4)
    ax.text(12, 5.1, 'Large:  25,668 params \u00d7 4 bytes  = 102,672 bytes  \u2248 100.3 KB',
            ha='center', fontsize=9, color=DARK, zorder=4)

    # Arrow down
    ax.annotate('', xy=(8, 4.2), xytext=(8, 4.7),
                arrowprops=dict(arrowstyle='->', color='#555', lw=3))

    # Formula
    formula_box = FancyBboxPatch((2.5, 2.8), 11, 1.3, boxstyle="round,pad=0.2",
                                 fc=LOSS_BG, ec=LOSS_BORDER, lw=2, zorder=3)
    ax.add_patch(formula_box)
    ax.text(8, 3.7, 'Compression Ratio  =  Original Size  /  Model Size',
            ha='center', fontsize=13, fontweight='bold', color=DARK, zorder=4)
    ax.text(8, 3.15, 'CR_base = 126,705,600 / 26,768  \u2248  4,734 : 1',
            ha='center', fontsize=11, fontweight='bold', color=ORANGE_FRONT, zorder=4)

    # Results bar chart style
    models_data = [
        ('Base\n6,692 params', 4734, BLUE_FRONT),
        ('Medium\n14,644 params', 2165, GREEN_FRONT),
        ('Large\n25,668 params', 1233, ORANGE_FRONT),
    ]
    bar_max = 5000
    for i, (label, cr, color) in enumerate(models_data):
        bx = 2.5 + i * 4.0
        bar_w = 3.2 * (cr / bar_max)
        bar = FancyBboxPatch((bx, 0.8), bar_w, 0.7,
                             boxstyle="round,pad=0.05", fc=color, ec='#333',
                             lw=1, zorder=3)
        ax.add_patch(bar)
        ax.text(bx + bar_w / 2, 1.15, '{:,} : 1'.format(cr),
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', zorder=4)
        ax.text(bx + bar_w / 2, 0.4, label, ha='center', va='top',
                fontsize=9, color=DARK)

    # Note
    ax.text(8, 2.3, '*Only target columns counted (input coordinates are query-time, not stored)',
            ha='center', fontsize=8, color='#888', fontstyle='italic')

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, 'compression_calc.png'), dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('Created: compression_calc.png')


if __name__ == '__main__':
    create_inr_architecture()
    create_model_comparison()
    create_online_diagram()
    create_comparison()
    create_compression()
    create_compression_calc()
    print('\nAll diagrams saved to:', OUT)
