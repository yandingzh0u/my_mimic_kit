"""Assemble Isaac Lab camera screenshots into an editable paper filmstrip.

Images remain actual raster captures, separately movable in the PowerPoint.
Only crop windows, typography, and whitespace are added by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'PAPER/artifacts/screenshots/mira_20261115'
OUT = ROOT / 'PAPER/figures'
NAME = 'mira_rollout_screenshots'
# Six chronological frames per row. The longer strips continue in a second
# row rather than shrinking twelve whole-body views into a single row.
MOTIONS = [('getup_facedown', 'Get-Up', 1.42), ('crawl', 'Crawl', 1.02),
           ('backflip', 'Backflip', 1.42), ('climb', 'Climb', 1.45),
           ('roll', 'Roll', 1.42), ('spinkick', 'Spin Kick', 1.42)]
DOUBLE_MOTIONS = {'backflip', 'climb', 'roll', 'spinkick'}
W, GAP, LABEL, RIGHT, ROW_GAP = 16.0, .040, 1.18, .08, .11
TILE_W = (W - LABEL - RIGHT - 5 * GAP) / 6
ROW_COUNT = sum(2 if key in DOUBLE_MOTIONS else 1 for key, _, _ in MOTIONS)
H = sum(h * (2 if key in DOUBLE_MOTIONS else 1) for key, _, h in MOTIONS) + ROW_COUNT * ROW_GAP + .06


def load():
    items = []
    for motion, label, height in MOTIONS:
        directory = DATA / motion
        meta = json.loads((directory / 'metadata.json').read_text())
        expected = 12 if motion in DOUBLE_MOTIONS else 6
        assert len(meta['frames']) == expected, (motion, len(meta['frames']))
        assert meta['actual_policy_rollout'] and meta['deterministic_mean_action']
        assert meta['training_updates'] == (3000 if motion == 'climb' else 2000)
        images, boxes = [], []
        for entry in meta['frames']:
            path = directory / entry['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            image = Image.open(path).convert('RGB')
            rgb = np.array(image).astype(float)
            mask = ((rgb[:, :, 1] > rgb[:, :, 0] + 25) &
                    (rgb[:, :, 2] > rgb[:, :, 0] + 25) &
                    (rgb[:, :, 1] > 65))
            yy, xx = np.nonzero(mask)
            assert len(xx) > 100, (motion, entry['file'])
            boxes.append([xx.min(), yy.min(), xx.max(), yy.max()])
            images.append((entry, path, image))
        # Compute one common crop from the union of the entire motion's body
        # silhouettes. Thus there is no frame-by-frame change of robot scale.
        boxes = np.array(boxes)
        x0, y0 = boxes[:, :2].min(axis=0)
        x1, y1 = boxes[:, 2:].max(axis=0)
        center_x, center_y = .5 * (x0 + x1), .5 * (y0 + y1)
        width_px, height_px = images[0][2].size
        aspect = TILE_W / height
        crop_w = max((x1 - x0) * 1.16, (y1 - y0) * 1.16 * aspect)
        # Keep a little more ledge around Climb's feet.
        if motion == 'climb':
            crop_w = max(crop_w, width_px * .96)
            center_y += 18
        crop_w = min(crop_w, width_px, height_px * aspect)
        crop_h = crop_w / aspect
        crop_x = max(0, min(width_px - crop_w, center_x - crop_w / 2))
        crop_y = max(0, min(height_px - crop_h, center_y - crop_h / 2))
        box = tuple(round(x) for x in (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        assert box[0] <= x0 and box[1] <= y0 and box[2] >= x1 and box[3] >= y1, (motion, box, (x0,y0,x1,y1))
        frames = []
        for entry, path, image in images:
            crop = image.crop(box)
            frames.append((entry, path, crop, box[0] / width_px, box[1] / height_px,
                           (width_px - box[2]) / width_px, (height_px - box[3]) / height_px))
        items.append((motion, label, height, meta, frames))
    return items


def ppt_text(slide, text, x, y, w, h, size, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = 'Arial'
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = RGBColor.from_string('263749')
    return shape


def build(items):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'font.size': 14})
    fig = plt.figure(figsize=(W, H), facecolor='white')
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.title = 'MIRA: actual Isaac Lab policy screenshots'
    prs.core_properties.subject = 'Actual RGB camera captures; editable layout and labels'
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    y = .05
    for motion, label, height, meta, frames in items:
        nrows = 2 if motion in DOUBLE_MOTIONS else 1
        total_h = nrows * height + (nrows - 1) * ROW_GAP
        fig.text(.04 / W, 1 - (y + total_h / 2) / H, label, ha='left', va='center',
                 fontsize=15.5, weight='bold', color='#263749')
        ppt_text(slide, label, .04, y + total_h / 2 - .2, LABEL - .08, .4, 15.5, True)
        for index, (entry, path, crop, left_crop, top_crop, right_crop, bottom_crop) in enumerate(frames):
            row, col = divmod(index, 6)
            xx = LABEL + col * (TILE_W + GAP)
            yy = y + row * (height + ROW_GAP)
            ax = fig.add_axes([xx / W, 1 - (yy + height) / H, TILE_W / W, height / H])
            ax.imshow(crop)
            ax.axis('off')
            ax.text(.045, .045, f"{entry['time_s']:.2f} s", transform=ax.transAxes,
                    color='#203044', fontsize=10.3, va='bottom', ha='left',
                    bbox={'facecolor': 'white', 'alpha': .84, 'edgecolor': 'none', 'pad': 2.0})
            pic = slide.shapes.add_picture(str(path), Inches(xx), Inches(yy),
                                           width=Inches(TILE_W), height=Inches(height))
            pic.crop_top, pic.crop_bottom = top_crop, bottom_crop
            pic.crop_left, pic.crop_right = left_crop, right_crop
            pic.name = f"{label}: simulator RGB at {entry['time_s']:.3f} seconds"
            caption = ppt_text(slide, f"{entry['time_s']:.2f} s", xx + .05,
                               yy + height - .25, .67, .20, 10.3)
            caption.fill.solid()
            caption.fill.fore_color.rgb = RGBColor(255, 255, 255)
        y += total_h + ROW_GAP
    slide.notes_slide.notes_text_frame.text = json.dumps(
        {'source': str(DATA.relative_to(ROOT)),
         'description': 'Real simulator RGB camera frames; each image remains a raster capture.',
         'layout_changes': 'Centered crop per motion, whitespace, labels only.',
         'provenance': [meta for _, _, _, meta, _ in items]}, indent=2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{NAME}.pdf', facecolor='white')
    fig.savefig(OUT / f'{NAME}.png', dpi=180, facecolor='white')
    plt.close(fig)
    prs.save(OUT / f'{NAME}.pptx')
    total_frames = sum(len(x[4]) for x in items)
    validation = {
        'actual_simulator': 'Isaac Lab / Isaac Sim RGB camera',
        'motion_order': [motion for motion, _, _, _, _ in items],
        'motions': [
            {'motion': motion, 'training_updates': meta['training_updates'],
             'physical_steps': meta['executed_steps'], 'screenshot_count': len(frames),
             'checkpoint_sha256': meta['model_sha256'],
             'capture_steps': [frame[0]['step'] for frame in frames],
             'rows': 2 if motion in DOUBLE_MOTIONS else 1,
             'common_crop_contains_all_robot_silhouettes': True}
            for motion, _, _, meta, frames in items],
        'total_frames': total_frames, 'row_count': ROW_COUNT,
        'slide_count': 1, 'embedded_movable_screenshots': total_frames,
        'slide_size_inches': [W, H],
        'source_script': 'PAPER/tools/capture_mira_screenshots.py',
        'assembly_script': 'PAPER/tools/build_mira_screenshot_figure.py',
        'image_integrity': 'Every official RGB screenshot is SHA-256 verified against capture metadata.',
        'files': {f'{NAME}.{ext}': hashlib.sha256((OUT / f'{NAME}.{ext}').read_bytes()).hexdigest()
                  for ext in ('png', 'pptx', 'pdf')}}
    (DATA / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(f'Wrote {NAME}: {total_frames} actual simulator frames; {W:.2f} × {H:.2f} inches')


if __name__ == '__main__':
    build(load())
