#!/usr/bin/env python3
"""Rebuild the four-panel MIRA reference as native PowerPoint artwork.

Run from any directory with python-pptx, Pillow, LibreOffice and PyMuPDF.
Paths/gradients are editable PowerPoint shapes; the robot renders are images.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

from mira_pipeline_panels import constraint, reward, scale


PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / 'figures'
ASSETS = FIGURES / 'assets'
OUTPUT_NAME = 'mira_pipeline_reference'
WIDTH, HEIGHT = 1672, 941
INK = '#111D37'
NAVY = '#122B62'
EDGE = '#4D6D9B'
TOP_FILL = '#FFFFFF'
CONNECTION = '#173260'
MATH = 'Liberation Serif'


def px(value):
    return round(value * 9144)


def pt(value):
    return Pt(value * .72)


def rgb(value):
    return RGBColor.from_string(value.lstrip('#').upper())


class Canvas:
    def __init__(self, slide, shapes=None):
        self.shapes = slide.shapes if shapes is None else shapes
        self.slide = slide

    def group(self, name):
        group = self.shapes.add_group_shape()
        group.name = name
        return Canvas(self.slide, group.shapes)

    @staticmethod
    def style(s, fill, line, lw):
        # Suppress theme default fills/lines/effects in both PowerPoint and PDF.
        style = s._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style')
        if style is not None:
            style.getparent().remove(style)
        s.shadow.inherit = False
        if fill is None:
            s.fill.background()
        elif isinstance(fill, tuple):
            s.fill.gradient()
            s.fill.gradient_angle = 270  # top to bottom
            for stop, color in zip(s.fill.gradient_stops, fill):
                stop.color.rgb = rgb(color)
        else:
            s.fill.solid()
            s.fill.fore_color.rgb = rgb(fill)
        if line is None:
            s.line.fill.background()
        else:
            s.line.color.rgb = rgb(line)
            s.line.width = pt(lw)
        return s

    def shape(self, x, y, w, h, fill=None, line=None, lw=1, r=0):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE
        s = self.shapes.add_shape(kind, px(x), px(y), px(w), px(h))
        if r:
            s.adjustments[0] = min(r / min(w, h), .5)
        return self.style(s, fill, line, lw)

    def oval(self, x, y, w, h, fill=None, line=None, lw=1):
        s = self.shapes.add_shape(MSO_SHAPE.OVAL, px(x), px(y), px(w), px(h))
        return self.style(s, fill, line, lw)

    def text(self, x, y, w, h, text, size=18, color=INK, bold=False,
             align='center', italic=False, font='Arial'):
        s = self.shapes.add_textbox(px(x), px(y), px(w), px(h))
        f = s.text_frame
        f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
        f.vertical_anchor = MSO_ANCHOR.MIDDLE
        f.word_wrap = False
        f.auto_size = MSO_AUTO_SIZE.NONE
        p = f.paragraphs[0]
        p.alignment = {'center': PP_ALIGN.CENTER, 'left': PP_ALIGN.LEFT,
                       'right': PP_ALIGN.RIGHT}[align]
        p.space_before = p.space_after = Pt(0)
        run = p.add_run()
        run.text = text
        run.font.name = font
        run.font.size = pt(size)
        run.font.color.rgb = rgb(color)
        run.font.bold = bold
        run.font.italic = italic
        return s

    def math(self, x, y, w, h, parts, size=22, color=INK):
        box = self.text(x, y, w, h, '', size, color, font=MATH)
        p = box.text_frame.paragraphs[0]
        p.clear()
        for text, italic, sub in parts:
            run = p.add_run()
            run.text = text
            run.font.name = MATH
            run.font.size = pt(size)
            run.font.color.rgb = rgb(color)
            run.font.italic = italic
            if sub:
                run._r.get_or_add_rPr().set('baseline', '-24000')
        return box

    def _path(self, points, fill, line, lw, close):
        path = self.shapes.build_freeform(px(points[0][0]), px(points[0][1]))
        path.add_line_segments([(px(x), px(y)) for x, y in points[1:]], close=close)
        return self.style(path.convert_to_shape(), fill, line, lw)

    def line(self, points, color=CONNECTION, width=1.2, arrow=False, dash=False):
        s = self._path(points, None, color, width, False)
        if dash:
            s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if arrow:
            end = OxmlElement('a:tailEnd')
            for k, v in {'type': 'triangle', 'w': 'med', 'len': 'med'}.items():
                end.set(k, v)
            s._element.spPr.get_or_add_ln().append(end)
        return s

    def poly(self, points, fill, line=None, lw=1):
        return self._path(points, fill, line, lw, True)

    def image(self, filename, x, y, w, h, name):
        with Image.open(filename) as im:
            iw, ih = im.size
        k = min(w / iw, h / ih)
        pic = self.shapes.add_picture(str(filename), px(x + (w - k * iw) / 2),
                                      px(y + (h - k * ih) / 2), px(k * iw), px(k * ih))
        pic.name = name
        return pic

    def check(self, cx, cy, r):
        self.oval(cx-r, cy-r, 2*r, 2*r, ('#79BB55', '#398A40'), '#408646', .8)
        self.line([(cx-r*.48, cy), (cx-r*.10, cy+r*.34),
                   (cx+r*.46, cy-r*.38)], '#FFFFFF', r*.22)

    def icon(self, name, cx, cy, sz, color):
        def pos(x, y):
            return cx + x * sz, cy + y * sz

        def draw(path, width=.075, arrow=False):
            return self.line([pos(*p) for p in path], color, width * sz, arrow)

        def dot(x, y, radius, fill=color, border=None):
            x, y = pos(x, y)
            r = radius * sz
            return self.oval(x-r, y-r, 2*r, 2*r, fill, border, sz*.052)

        def arc(a0, a1, rx, ry, x=0., y=0., arrow=False):
            angles = [math.radians(a0 + (a1-a0)*i/22) for i in range(23)]
            draw([(x+rx*math.cos(a), y+ry*math.sin(a)) for a in angles],
                 .063, arrow)

        if name == 'pin':
            angles = [math.radians(158+i*224/30) for i in range(31)]
            ps = [(0, .46)] + [(.31*math.cos(a), -.14+.31*math.sin(a)) for a in angles]
            self.poly([pos(*p) for p in ps], color)
            dot(0, -.12, .113, '#FFFFFF')
        elif name in ('rotation', 'angular'):
            arc(25, 170, .39, .36, arrow=True)
            arc(200, 352, .39, .36, arrow=True)
            if name == 'angular':
                draw([(0, .18), (0, -.18), (.1, -.28)], .052)
        elif name in ('nodes', 'policy'):
            points = [(-.36,.18), (-.02,-.29), (.34,.06), (.05,.25)]
            for a, b in ((0,1), (1,2), (2,3), (3,0), (1,3)):
                draw([points[a], points[b]], .047)
            for x, y in points:
                dot(x, y, .09, '#F2F7FD', color)
        elif name == 'body':
            dot(0, -.31, .08)
            for p in [[(0,-.20),(0,.11)], [(-.29,-.09),(0,-.15),(.29,-.09)],
                      [(-.21,.36),(0,.09),(.21,.36)]]:
                draw(p, .083)
        elif name in ('gauge', 'velocity'):
            if name == 'gauge':
                arc(163, 378, .39, .37, y=.055)
                draw([(-.36,.17),(.36,.17)], .058)
                for a in (197, 245, 295, 342):
                    t = math.radians(a)
                    draw([(.27*math.cos(t),.055+.27*math.sin(t)),
                          (.35*math.cos(t),.055+.35*math.sin(t))], .035)
                draw([(0,.065),(.195,-.19)], .058)
                dot(0, .065, .057)
            else:
                for i, (x1, x2) in enumerate([(-.35,.25),(-.44,.38),(-.29,.16)]):
                    yy = -.22 + i*.22
                    draw([(x1,yy),(x2,yy)], .071, i == 1)
        elif name == 'gear':
            outline = []
            for j in range(8):
                for off, rad in ((-.50,.32), (-.30,.43), (.20,.43), (.40,.32)):
                    a = (j + off) * math.tau / 8
                    outline.append(pos(rad*math.cos(a), rad*math.sin(a)))
            self.poly(outline, '#B8C9DE', color, .047*sz)
            dot(0, 0, .14, '#EDF3FA', color)
        else:
            raise ValueError(f'Unknown icon: {name}')


def frame(d, x, y, w, h, title, tab_width=None, size=21):
    d.shape(x, y, w, h, TOP_FILL, EDGE, 1.35, 11)
    tab_width = tab_width or w
    d.shape(x+.65, y+.65, tab_width-1.3, 40, ('#163D8B', '#0C245F'), r=10)
    d.shape(x+.65, y+27, tab_width-1.3, 13.7, ('#113172', '#0C245F'))
    d.text(x+14, y+2, tab_width-25, 35, title, size, '#FFFFFF', True, 'left')


def overview(d, robot_files, robot_names):
    frame(d, 10, 10, 1654, 383, '(a)  Overview of MIRA reward learning', 521, 25)
    for y, title, state, image, name, (a, b, edge, color) in zip(
        (65,190), ('Reference state', 'Simulated state'), ('ŝ','s'),
        robot_files, robot_names,
        (('#F3FAED','#E1F1D9','#A5C893','#234829'),
         ('#FFF5EB','#F9E6D8','#E1B58D','#6E3B23')),
    ):
        d.shape(35, y, 225, 109, (a,b), edge, 1.3, 12)
        d.image(image, 48, y+6, 88, 97, name)
        d.text(137, y+24, 110, 25, title, 15.8, color, True)
        d.math(144, y+48, 96, 34, [(state, True, False), ('t', True, True)], 25, color)

    d.shape(311, 58, 255, 243, ('#F5FAFF','#E3F0FE'), '#83A8D1', 1.35, 12)
    d.shape(311.8, 58.8, 253.4, 34, ('#D6E9FD','#BDD8F2'), r=10)
    d.shape(312, 79, 253, 14, ('#C9E0F7','#BDD8F2'))
    d.text(322, 61, 205, 27, 'Physical error groups', 18.5, INK, True)
    d.math(521, 61, 36, 27, [('e', True, False), ('t', True, True)], 22)
    for y, glyph, label in zip(
        range(104, 269, 25), ('pin','rotation','nodes','body','gauge','velocity','angular'),
        ('Root position','Root rotation','Body position','Body rotation','Joint velocity',
         'Root linear velocity','Root angular velocity'),
    ):
        d.icon(glyph, 340, y+1, 20, '#285B9B')
        d.text(363, y-10, 184, 21, label, 15.8, INK, True, 'left')
        if y < 245:
            d.line([(330, y+13), (549, y+13)], '#CEDFF0', .7)
    d.text(322, 273, 233, 22, '(robot-dependent dimensions)', 14, '#4B6482', italic=True)

    d.shape(603, 132, 184, 107, ('#EDF7FF','#D2E6FB'), '#7D9EC9', 1.35, 11)
    d.text(612, 142, 166, 23, 'Group', 20.5, INK, True)
    d.text(611, 166, 168, 23, 'Normalization', 20.5, INK, True)
    d.math(613, 195, 164, 34, [('z', True, False), ('t', True, True)], 26)

    for y, heading, is_zero, colors in [
        (103,'Positive Sample',True,('#F3FBEA','#E5F4DC','#8FBB76','#2C5429')),
        (207,'Negative Sample',False,('#FFF7EF','#FBE9DA','#D8AA81','#743921')),
    ]:
        a,b,edge,col=colors
        d.shape(823, y, 170, 72, (a,b), edge, 1.35, 10)
        d.text(829, y+7, 158, 23, heading, 17, col, True)
        d.math(829, y+32, 158, 30,
               [('0', False, False)] if is_zero else [('z', True, False), ('t', True, True)],
               26, col)

    d.shape(1039, 119, 336, 158, ('#F3F9FE','#DEECFA'), '#779CC7', 1.35, 12)
    d.text(1050, 124, 314, 28, 'MIRA Discriminator', 22, INK, True)
    d.line([(1051,153),(1363,153)], '#A5BFDD', .9)
    # Conceptual network glyph. Node count is illustrative.
    layers = [[(1065,y) for y in (170,188,206,224)],
              [(1104,y) for y in (167,182,197,212,227)],
              [(1144,y) for y in (173,190,207,224)], [(1180,198)]]
    for left, right in zip(layers, layers[1:]):
        for start in left:
            for end in right:
                d.line([start, end], '#8DAFD4', .78)
    for i, layer in enumerate(layers):
        for cx,cy in layer:
            d.oval(cx-4.7, cy-4.7, 9.4,9.4,
                   ('#DCECFB','#8EB8E7') if i<3 else ('#BAE1A1','#6FA65B'),
                   '#4B7BAF' if i<3 else '#57894C', .8)
    for y, formula in ((162,'W ≥ 0'),(206,'‖W‖₂ ≤ 1')):
        d.shape(1210,y,146,34,('#F5FBEF','#E3F2D9'),'#95BC7D',1.05,8)
        d.check(1229,y+17,9.8)
        d.text(1242,y+1,103,32,formula,19,INK,True,font=MATH)
    d.text(1052, 243, 309, 25, 'Monotone & 1-Lipschitz', 17.5, '#1F3B69', True)

    d.shape(1411, 147, 227, 86, ('#F1F8FF','#CEE4FA'), '#7C9DC4', 1.35, 12)
    d.text(1424, 154, 200, 30, 'Imitation Reward', 21, INK, True)
    d.math(1423, 188, 201, 34, [('r', True, False), ('t', True, True)], 27)

    for x,w in [(328,295),(776,216),(1202,205)]:
        d.shape(x,331,w,48,('#F5F8FC','#E1EAF4'),'#92A8C3',1.2,11)
    d.icon('gear', 359,355,34,'#466994')
    d.text(384,337,223,35,'Physics simulation',21,INK,True)
    d.icon('policy', 807,355,33,'#3F6498')
    d.text(833,337,96,35,'Policy',21,INK,True)
    d.math(922,338,50,32,[('π',True,False)],24)
    d.text(1210,337,189,35,'PPO',22,INK,True)

    # Both observations contribute to the measured error. The ideal zero
    # sample is independent of the normalized error (negative-sample branch).
    d.line([(260,120),(284,120),(287,123),(287,244),(284,247),(260,247)],
           CONNECTION,2.05)
    d.line([(287,185),(311,185)], CONNECTION,2.05, True)
    d.line([(566,190),(603,190)],CONNECTION,2.05,True)
    d.line([(787,190),(800,190),(800,241),(803,244),(823,244)],CONNECTION,2.05,True)
    d.line([(993,138),(1010,138),(1015,143),(1015,238),(1010,243),(993,243)],
           CONNECTION,2.05)
    d.line([(1015,194),(1039,194)],CONNECTION,2.05, True)
    d.line([(1375,190),(1411,190)],CONNECTION,2.05,True)
    d.line([(1545,233),(1545,349),(1540,355),(1407,355)],CONNECTION,2.05,True)
    d.line([(1202,355),(992,355)],'#785590',2,True,True)
    d.text(1029,326,137,24,'Update π',16,'#694A80',True)
    d.line([(776,355),(623,355)],CONNECTION,2.05,True)
    d.math(653,326,70,24,[('a',True,False),('t',True,True)],20)
    d.line([(328,355),(148,355),(141,348),(141,299)],CONNECTION,2.05,True)


def build(output_dir):
    folder = ASSETS / 'isaac_humanoid'
    images = [folder / f'{name}.png' for name in ('reference', 'simulated')]
    names = ('Paper humanoid in Isaac Lab • reference motion',
             'Paper humanoid in Isaac Lab • simulated motion')
    for filename in images:
        if not filename.is_file():
            raise FileNotFoundError(filename)

    prs = Presentation()
    prs.slide_width, prs.slide_height = px(WIDTH), px(HEIGHT)
    prs.core_properties.title = 'MIRA: Overview of reward learning'
    prs.core_properties.subject = 'Editable recreation of the four-panel method figure'
    prs.core_properties.author = 'MIRA'
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb('#FFFFFF')
    d = Canvas(slide)
    overview(d.group('a · Overview of MIRA reward learning'), images, names)
    for name, x, w, title, size, module in [
        ('b',10,487,'Scale balancing in the physical error space',20.5,scale),
        ('c',510,565,'Constrained monotone discriminator',22,constraint),
        ('d',1087,579,'Reward geometry in error space',22,reward),
    ]:
        panel = d.group(f'{name} · {title}')
        frame(panel, x, 412, w, 510, f'({name})  {title}', size=size)
        module.draw(panel)
    slide.notes_slide.notes_text_frame.text = (
        'Both robot images use the Isaac Lab humanoid from the paper experiments, '
        'captured at the same Backflip timestep. See figures/assets/isaac_humanoid/provenance.json.\n'
        'The lower panels use illustrative values, not measured results. '
        'The scale plot assumes equal normalized per-component errors. The weight example '
        'uses elementwise division for t and full precision until display. The reward '
        'contour illustrates a monotone softplus reward. All non-robot artwork is native '
        'PowerPoint text or geometry. Select a panel and use Ungroup to edit individual objects.'
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ppt = output_dir / f'{OUTPUT_NAME}.pptx'
    prs.save(ppt)
    with tempfile.TemporaryDirectory(prefix='mira_pipeline_reference_') as tmp:
        tmp = Path(tmp)
        subprocess.run(['libreoffice', '-env:UserInstallation='+(tmp/'lo').as_uri(),
                        '--headless', '--convert-to', 'pdf', '--outdir', str(tmp), str(ppt)],
                       check=True, capture_output=True, text=True)
        import fitz
        with fitz.open(tmp/f'{OUTPUT_NAME}.pdf') as doc:
            assert len(doc) == 1
            doc.save(output_dir/f'{OUTPUT_NAME}.pdf', garbage=4, deflate=True)
            doc[0].get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(
                output_dir/f'{OUTPUT_NAME}.png')
    print(ppt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=FIGURES)
    args = parser.parse_args()
    build(args.output_dir.resolve())


if __name__ == '__main__':
    main()
