"""Publication figures and a native, editable PowerPoint from measured seed-0 logs.

Reads the audited manifest produced by build_mira_experiments.py. Does not synthesize
seeds, uncertainty, missing results, motion images, or independent evaluations.
The PowerPoint uses native editable charts, with no tables, heatmaps or screenshots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_TICK_MARK, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'PAPER'
FIGURES = PAPER / 'figures'
MANIFEST = PAPER / 'generated/mira_measured_seed0.json'
MOTIONS = ('roll', 'climb', 'backflip', 'crawl', 'getup_facedown', 'spinkick')
MOTION_NAMES = ('Roll', 'Climb', 'Backflip', 'Crawl', 'Get-Up', 'Spin Kick')
METHODS = ('AMP', 'DeepMimic', 'ADD', 'MIRA')
METRICS = (
    ('Root_Pos_Err', 'Root position', 'm'),
    ('Root_Rot_Err', 'Root rotation', 'rad'),
    ('Body_Pos_Err', 'Body position', 'm'),
    ('Body_Rot_Err', 'Body rotation', 'rad'),
    ('Dof_Vel_Err', 'Joint velocity', 'rad/s'),
    ('Root_Vel_Err', 'Root velocity', 'm/s'),
    ('Root_Ang_Vel_Err', 'Root angular velocity', 'rad/s'),
)
TRACK = (METRICS[2], METRICS[3], METRICS[4])
VARIANTS = ('MIRA', 'w/o Monotone Modeling', 'w/o Diagonal Norm.', 'w/o Both Components')
LABELS = ('MIRA', 'w/o monotone modeling', 'w/o diagonal normalization', 'w/o both')
COLORS = {'AMP': '#A2AAB4', 'DeepMimic': '#617D9B', 'ADD': '#C58A3D', 'MIRA': '#00887D',
          VARIANTS[1]: '#526F96', VARIANTS[2]: '#C58A3D', VARIANTS[3]: '#906889'}
NAVY = '#17334E'
MUTED = '#586877'
GRID = np.array([1, *range(101, 2000, 100), 2000], dtype=float)
PROGRESS_GRID = GRID / 2000
SAMPLES_PER_UPDATE = 8192 * 32


def json_lines(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def load_data() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('actual_seed_ids') != [0]:
        raise ValueError('Figure generator requires an explicitly measured seed-0 manifest.')
    for method in METHODS:
        for motion in MOTIONS:
            run = manifest['runs'][method][motion]
            if not isinstance(run, dict):
                raise ValueError(f'Missing measured result: {method}/{motion}')
            expected_budget = 3000 if method == 'MIRA' and motion == 'climb' else 2000
            if run['completed_updates_used'] != expected_budget:
                raise ValueError(f'Unexpected recorded budget: {method}/{motion}')
            run['_rows'] = json_lines(run['path'])
    for name, run in manifest['ablations'].items():
        run['_rows'] = json_lines(run['path'])
        run['_pressure'] = json_lines(run['pressure_source'])
    return manifest


def measured_curve(run: dict, key: str, relative_progress: bool = False) -> np.ndarray:
    budget = run['completed_updates_used']
    target = PROGRESS_GRID * budget if relative_progress else GRID
    by_update = {}
    for r in run['_rows']:
        update = float(r.get('Samples', r.get('samples'))) / SAMPLES_PER_UPDATE
        if 0 < update <= budget and key in r:
            by_update[update] = float(r[key])
    xs = np.array(sorted(by_update))
    if len(xs) < 2 or xs[0] > target[0] or xs[-1] < target[-1]:
        raise ValueError(f'Incomplete measured curve: {run["path"]}: {key}')
    return np.interp(target, xs, [by_update[x] for x in xs])


def macro_curve(data: dict, method: str, key: str) -> np.ndarray:
    return np.mean([measured_curve(data['runs'][method][motion], key, relative_progress=True) for motion in MOTIONS], axis=0)


def style() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'axes.titleweight': 'normal',
        'axes.edgecolor': '#B2BDC6', 'axes.labelcolor': NAVY, 'text.color': NAVY,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'grid.color': '#DDE3E8', 'grid.linewidth': 0.45,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(FIGURES / f'{name}.{extension}', dpi=260, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)


def line_axes(ax, title: str, ylabel: str, progress: bool = False) -> None:
    ax.set_title(title, loc='left', pad=7)
    ax.set_xlabel('Training progress (%)' if progress else 'PPO updates')
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_xlim(0, 100 if progress else 2000)
    ax.set_xticks([0, 25, 50, 75, 100] if progress else [0, 500, 1000, 1500, 2000])
    ax.grid(axis='y')
    ax.tick_params(length=2.5, width=.6, pad=2)


def logarithmic_bounds(values):
    lower = float(.1 * 2 ** np.floor(np.log2(np.min(values) / .1)))
    steps = int(np.ceil(np.log2(np.max(values) / lower)))
    return lower, float(lower * 2 ** steps), [lower * 2 ** k for k in range(steps + 1)]


def plot_tracking(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.38))
    for index, (ax, (key, name, unit)) in enumerate(zip(axes, TRACK)):
        for method in METHODS:
            ax.plot(PROGRESS_GRID * 100, macro_curve(data, method, key), color=COLORS[method],
                    lw=1.65, label=method, zorder=2)
        line_axes(ax, f'({chr(97 + index)}) {name}', f'Error ({unit}; log scale)', progress=True)
        ax.set_yscale('log')
        all_values = np.concatenate([macro_curve(data, method, key) for method in METHODS])
        lower, upper, ticks = logarithmic_bounds(all_values)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, 1.025),
               ncol=4, frameon=False, handlelength=2.2, columnspacing=2.2, fontsize=8)
    fig.subplots_adjust(left=.065, right=.985, top=.8, bottom=.22, wspace=.43)
    save(fig, 'mira_tracking_curves')


def diagnostic_points(run: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    if key == 'Disc_Neg_Logit':
        return GRID, measured_curve(run, key)
    dedup = {int(r['completed_updates']): float(r[key]) for r in run['_pressure']
             if 0 < int(r['completed_updates']) <= 2000}
    xs = np.array(sorted(dedup), dtype=float)
    ys = np.array([dedup[x] for x in xs])
    if key == 'sampled_reward_exact_zero_fraction':
        ys *= 100
    return xs, ys


def plot_diagnostics(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.52))
    panels = (('maximum_score_gradient_L2', 'Score sensitivity', r'Max. sampled $\|\nabla_z s\|_2$'),
              ('Disc_Neg_Logit', 'Policy-sample score', 'Mean negative logit'),
              ('sampled_reward_exact_zero_fraction', 'Numerically zero rewards', 'Sampled zero rewards (%)'))
    for i, (ax, (key, name, ylabel)) in enumerate(zip(axes, panels)):
        for variant, label in zip(VARIANTS, LABELS):
            x, y = diagnostic_points(data['ablations'][variant], key)
            ax.plot(x, y, lw=1.9 if variant == 'MIRA' else 1.3,
                    color=COLORS[variant], label=label,
                    zorder=5 if variant == 'MIRA' else 2)
        line_axes(ax, f'({chr(97+i)}) {name}', ylabel)
        if i == 0:
            ax.axhline(1, color=NAVY, linestyle=(0, (3, 3)), lw=.8, alpha=.65, zorder=1)
            ax.set_ylim(bottom=0)
        if i == 2:
            ax.set_ylim(-1.5, max(42, ax.get_ylim()[1]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, 1.04),
               ncol=2, frameon=False, handlelength=2, columnspacing=2.7, fontsize=7.5)
    fig.subplots_adjust(left=.07, right=.99, top=.73, bottom=.2, wspace=.48)
    save(fig, 'mira_discriminator_diagnostics')


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value.lstrip('#'))


def textbox(slide, text, left, top, width, height, size=16, bold=False,
            color=NAVY, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = shape.text_frame
    tf.clear()
    tf.margin_left = Inches(.02)
    tf.margin_right = Inches(.02)
    tf.margin_top = 0
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.name = 'Aptos'
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = rgb(color)
    return shape


def slide_base(prs, title: str, subtitle: str, index: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bar = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, prs.slide_width, Inches(.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(COLORS['MIRA']); bar.line.fill.background()
    textbox(slide, title, .55, .42, 12.2, .48, size=27, bold=True)
    textbox(slide, subtitle, .58, 1.02, 12.1, .56, size=13, color=MUTED)
    textbox(slide, 'MIRA  |  measured seed 0', .58, 7.11, 11, .2, size=10, color=MUTED)
    textbox(slide, str(index), 12.2, 7.08, .5, .23, size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    return slide


def xy_chart(slide, series, left, top, width, height, title: str, ylabel: str,
             y_zero: bool = False, log_y: bool = False, progress: bool = False):
    chart_data = XyChartData()
    for name, x, y, color in series:
        s = chart_data.add_series(name)
        for xx, yy in zip(x, y):
            s.add_data_point(float(xx), float(yy))
    chart = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS,
        Inches(left), Inches(top), Inches(width), Inches(height), chart_data).chart
    chart.has_title = False
    chart.has_legend = False
    chart.chart_style = 13
    chart.font.name = 'Aptos'
    chart.font.size = Pt(11)
    for target, (_, _, _, color) in zip(chart.series, series):
        target.format.line.color.rgb = rgb(color)
        target.format.line.width = Pt(1.8)
    xaxis = chart.category_axis
    xaxis.minimum_scale = 0
    xaxis.maximum_scale = 100 if progress else 2000
    xaxis.major_unit = 25 if progress else 500
    xaxis.has_title = True
    xaxis.axis_title.text_frame.text = 'Training progress (%)' if progress else 'PPO updates'
    xaxis.axis_title.text_frame.paragraphs[0].font.size = Pt(11)
    xaxis.tick_labels.font.size = Pt(10)
    xaxis.major_tick_mark = XL_TICK_MARK.NONE
    xaxis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    yaxis = chart.value_axis
    yaxis.has_title = True
    yaxis.axis_title.text_frame.text = ylabel
    yaxis.axis_title.text_frame.paragraphs[0].font.size = Pt(11)
    yaxis.tick_labels.font.size = Pt(10)
    yaxis.tick_labels.number_format = '0.###' if log_y else '0.0'
    yaxis.major_tick_mark = XL_TICK_MARK.NONE
    yaxis.has_major_gridlines = True
    yaxis.major_gridlines.format.line.color.rgb = rgb('#E1E7EB')
    yaxis.major_gridlines.format.line.width = Pt(.6)
    if y_zero:
        yaxis.minimum_scale = 0
    if log_y:
        values = np.concatenate([np.array(y) for _, _, y, _ in series])
        lower, upper, _ = logarithmic_bounds(values)
        yaxis.minimum_scale = lower
        yaxis.maximum_scale = upper
        scaling = yaxis._element.find(qn('c:scaling'))
        logbase = OxmlElement('c:logBase'); logbase.set('val', '2'); scaling.insert(0, logbase)
        yaxis.crosses_at = yaxis.minimum_scale
    elif not y_zero:
        values = np.concatenate([np.array(y) for _, _, y, _ in series])
        if values.min() < 0:
            yaxis.minimum_scale = float(5 * np.floor(values.min() / 5))
            yaxis.maximum_scale = max(0, float(5 * np.ceil(values.max() / 5)))
            yaxis.crosses_at = yaxis.minimum_scale
    textbox(slide, title, left+.15, top-.38, width-.3, .33, size=16, bold=True)
    return chart


def legend(slide, names, left, top, total_width, labels=None):
    widths = total_width / len(names)
    for i, name in enumerate(names):
        x = left + i * widths
        line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,
                                      Inches(x), Inches(top+.085), Inches(.27), Inches(.035))
        line.fill.solid(); line.fill.fore_color.rgb = rgb(COLORS[name]); line.line.fill.background()
        inherited = line._element.find(qn('p:style'))
        if inherited is not None: line._element.remove(inherited)
        textbox(slide, labels[i] if labels else name, x+.35, top, widths-.38, .25, size=11)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def create_ppt(data: dict) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    properties = prs.core_properties
    properties.title = 'MIRA — Measured Experimental Results'
    properties.subject = 'Editable learning curves and Backflip discriminator diagnostics'
    properties.author = 'MIRA paper'
    s = slide_base(prs, 'Tracking performance across six motions',
        'Equal-weight means across Roll, Climb, Backflip, Crawl, Get-Up and Spin Kick; no synthetic seed bands.', 1)
    legend(s, METHODS, 4, 1.6, 6)
    for i, (key, name, unit) in enumerate(TRACK):
        series = [(m, PROGRESS_GRID * 100, macro_curve(data, m, key), COLORS[m]) for m in METHODS]
        xy_chart(s, series, .45+i*4.28, 2.25, 4.25, 3.85, name, f'Error ({unit}; log scale)', log_y=True, progress=True)
    textbox(s, 'Training progress: MIRA Climb uses 3000 updates; all other runs use 2000. Errors cover executed test timesteps.',
            .65, 6.45, 12, .4, size=12, color=MUTED)
    add_notes(s, 'Source: PAPER/generated/mira_measured_seed0.json and the train_metrics.jsonl paths recorded there. '
        'All six motions, equal motion weights, one measured seed (0). MIRA Climb uses 3000 updates, all other runs 2000. Each curve aligns fractions of the actual training budget, not equal absolute samples. '
        'Curves interpolate only between actual logged test snapshots; no smoothing or fabricated uncertainty. '
        'These are training-time reference-start tests, not newly generated independent evaluations. '
        'All charts, data series, titles and text remain editable in PowerPoint.')

    s = slide_base(prs, 'Discriminator behavior during the Backflip ablation',
        'Backflip, 2000 updates: sensitivity in the normalized discriminator-input space.', 2)
    legend(s, VARIANTS, .75, 1.68, 12, labels=['MIRA', 'w/o monotone model', 'w/o diagonal norm.', 'w/o both'])
    paneldefs = (('maximum_score_gradient_L2','Sampled score sensitivity','Max. gradient L2',True),
                 ('Disc_Neg_Logit','Policy-sample score','Mean negative logit',False),
                 ('sampled_reward_exact_zero_fraction','Numerically zero rewards','Sampled zero rewards (%)',True))
    for i,(key,title,label,zero) in enumerate(paneldefs):
        series=[]
        for variant in VARIANTS:
            x,y=diagnostic_points(data['ablations'][variant], key)
            series.append((variant,x,y,COLORS[variant]))
        xy_chart(s, series, .45+i*4.28, 2.35, 4.25, 3.85, title, label,zero)
    textbox(s, 'Sensitivity probes use each variant’s own normalized input. Zero reward means exact numerical zero in sampled rollout diagnostics.',
            .65, 6.48, 12, .4, size=12, color=MUTED)
    add_notes(s, 'Sources: constraint_rounds.jsonl and train_metrics.jsonl recorded in the measured manifest. '
        'Gradient values are sampled maxima, not measured global Lipschitz constants; the constrained architecture has a separate analytic bound. '
        'The no-monotone variant has 172 normalized signed-coordinate inputs and the physical variants 64 normalized physical-error inputs. '
        'Cross-variant gradients are therefore not derivatives with respect to identical physical coordinates. '
        'Exact-zero fraction describes floating-point reward evaluation, not a proof that normalized PPO advantages disappear. '
        'No new physical intervention or independent evaluation was performed to create these charts.')

    prs.save(FIGURES/'mira_results_editable.pptx')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-pending',action='store_true',help='Check source syntax only if any six-motion data are pending.')
    args=parser.parse_args()
    try:
        data=load_data()
    except ValueError:
        if args.allow_pending:
            print('Six-motion real data pending; no figures overwritten.')
            return
        raise
    style()
    plot_tracking(data)
    plot_diagnostics(data)
    create_ppt(data)
    for obsolete in ('mira_motion_summary', 'mira_ablation_effects', 'mira_error_comparison'):
        for extension in ('pdf', 'png'):
            (FIGURES / f'{obsolete}.{extension}').unlink(missing_ok=True)
    print('Saved learning and diagnostic figures (PDF+PNG) and two-slide editable PowerPoint; no redundant error table plot.')


if __name__ == '__main__':
    main()
