"""Editable PowerPoint drawing for MIRA overview panel (d)."""

from math import atan2, cos, hypot, pi, sin

from pptx.dml.color import RGBColor
from pptx.util import Pt


NAVY = '#101A39'
TEXT = '#10162C'
L_BLUE = '#DDEBFA'
GREEN = '#70A963'
GREEN_TEXT = '#164918'
ORANGE = '#F6A12C'
ORANGE_TEXT = '#9D3919'
WHITE = '#FFFFFF'


def _solid(hexstr):
    return RGBColor.from_string(hexstr.lstrip('#').upper())


def _gradient(shape, start, end, angle=0):
    """Native two-stop DrawingML gradient on an editable native shape."""
    shape.fill.gradient()
    shape.fill.gradient_angle = angle
    for stop, color in zip(shape.fill.gradient_stops, (start, end)):
        stop.color.rgb = _solid(color)


def _math(d, x, y, w, h, tokens, size=17, color=TEXT, align='center'):
    """Native rich text math; tuple tokens specify subscripts and italics."""
    shape = d.text(x, y, w, h, '', size=size, color=color, align=align, font='Georgia')
    paragraph = shape.text_frame.paragraphs[0]
    paragraph.clear()
    for item in tokens:
        if isinstance(item, str):
            chars, subscript, italic = item, False, False
        else:
            chars, subscript, italic = item
        run = paragraph.add_run()
        run.text = chars
        run.font.name = 'Georgia'
        run.font.size = Pt(size * .72)
        run.font.color.rgb = _solid(color)
        run.font.italic = italic
        if subscript:
            run._r.get_or_add_rPr().set('baseline', '-22500')
    return shape


def _m(symbol):
    return (symbol, False, True)


def _sub(symbol):
    return (symbol, True, True)


def _score(var):
    return [_m('s'), _sub('θ'), '(', _m(var), ')']


def _energy(var):
    return [_m('E'), _sub('θ'), '(', _m(var) if var != '0' else '0', ')']


def _reward(point):
    return [_m('r'), '(', _m('z'), _sub(point), ')']


def _palette_at(u):
    # Light blue for large error, deep blue for small error. Interpolated
    # palette stops are used to produce soft, editable contour bands.
    anchors = [
        (0.00, (5, 62, 145)),
        (0.12, (11, 76, 174)),
        (0.30, (22, 109, 211)),
        (0.52, (68, 143, 224)),
        (0.72, (116, 170, 233)),
        (0.88, (164, 199, 240)),
        (1.00, (213, 230, 249)),
    ]
    for (u0, a), (u1, b) in zip(anchors, anchors[1:]):
        if u <= u1:
            t = max(0, (u - u0) / (u1 - u0))
            return '#' + ''.join(f'{round(aa + t * (bb - aa)):02X}' for aa, bb in zip(a, b))
    return '#D5E6F9'


def _contours(d, left, top, right, bottom):
    # Schematic of softplus(b - sqrt((x/a)^2 + (y/c)^2)). Each shape is
    # its super-level set clipped to the chart rectangle, restricted to
    # the nonnegative quadrant. The figure does not depict measured rewards.
    w, h = right - left, bottom - top
    ax, ay = w * 1.03, h * 0.81
    limit = hypot(w / ax, h / ay)
    d.shape(left, top, w, h, _palette_at(1.0))
    n = 25
    corner = atan2(h / ay, w / ax)
    angles = sorted({i * pi / 96 for i in range(49)} | {corner})
    for i in range(n - 1, 0, -1):
        radius = limit * i / n
        points = [(left, bottom)]
        for theta in angles:
            dx, dy = ax * cos(theta), ay * sin(theta)
            clip_x = w / dx if dx > 1e-9 else float('inf')
            clip_y = h / dy if dy > 1e-9 else float('inf')
            r = min(radius, clip_x, clip_y)
            points.append((min(right, left + r * dx), max(top, bottom - r * dy)))
        d.poly(points, _palette_at((i - .5) / n))


def _point(d, x, y, diameter, fill):
    d.oval(x - diameter / 2, y - diameter / 2, diameter, diameter, fill, WHITE, 1.7)


def _label(d, x, y, point):
    d.shape(x, y, 41, 26, '#FFF1DF', '#D98D46', 1.1, r=7)
    _math(d, x + 2, y + 1, 37, 24, [_m('z'), _sub(point)], 19, ORANGE_TEXT)


def draw(d):
    left, top, right, bottom = 1130, 484, 1425, 793
    _contours(d, left, top, right, bottom)
    d.line([(1129, 793), (1444, 793)], NAVY, 2.1, arrow=True)
    d.line([(1129, 793), (1129, 470)], NAVY, 2.1, arrow=True)

    d.text(1144, 799, 265, 24, 'root pos. error', 16.0, TEXT, bold=True)
    ylabel = d.text(977, 620, 260, 24, 'Normalized root rotation error', 16.0, TEXT, bold=True)
    ylabel.rotation = 270

    legend = d.shape(1142, 489, 171, 55, L_BLUE, '#92A9C7', 1.1, r=6)
    _gradient(legend, '#EDF4FE', '#DAE7F7', 270)
    _point(d, 1159, 503, 14, '#43AE2B')
    _point(d, 1159, 528, 14, ORANGE)
    d.text(1173, 493, 133, 19, 'Ideal sample', 13.5, TEXT, bold=True, align='left')
    d.text(1173, 518, 133, 19, 'Policy samples', 13.5, TEXT, bold=True, align='left')
    d.text(1324, 488, 99, 19, '2D example', 14, '#455367', italic=True)

    # The two perpendicular segments expose the comparison coordinate by
    # coordinate. They are guides, not a trajectory or gradient direction.
    guide = d.line([(1244, 692), (1331, 692), (1331, 607)], WHITE, 2.6, dash=True)
    guide.name = 'Componentwise order guide between z_a and z_b'
    for x, y, diameter in [
        (1180, 569, 12), (1257, 605, 13), (1331, 607, 17),
        (1366, 688, 13), (1244, 692, 17), (1301, 730, 13),
    ]:
        _point(d, x, y, diameter, ORANGE)
    _point(d, 1131, 793, 16, '#43AE2B')
    _label(d, 1221, 655, 'a')
    _label(d, 1334, 572, 'b')

    d.poly([(1142, 785), (1161, 766), (1169, 766)], '#C4E9A0')
    d.shape(1150, 745, 90, 25, '#E6F7CE', GREEN, 1.1, r=5)
    d.text(1155, 747, 80, 21, 'zero error', 14, GREEN_TEXT, bold=True)

    definition = d.shape(1434, 513, 222, 144, '#EAF3FC', '#A9C4DD', 1.15, r=12)
    _gradient(definition, '#F0F7FD', '#D2E7FB', 270)
    d.text(1446, 518, 198, 27, 'Reward Definition', 18.5, NAVY, bold=True)
    d.shape(1443, 554, 205, 42, '#FAFCFF', '#A9BDD2', 1.05, r=6)
    _math(d, 1445, 555, 201, 41, _score('z') + [
        ' = ', _m('b'), ' − [', *_energy('z'), ' − ', *_energy('0'), ']'
    ], 16.0)
    d.shape(1443, 604, 205, 42, '#FAFCFF', '#A9BDD2', 1.05, r=6)
    _math(d, 1448, 605, 195, 41, [
        _m('r'), _sub('θ'), '(', _m('z'), ') = softplus(', *_score('z'), ')'
    ], 16.0)

    inequality = d.shape(1438, 678, 218, 82, '#EBF7E0', GREEN, 1.25, r=9)
    _gradient(inequality, '#F5FBED', '#E6F4D9', 270)
    _math(d, 1450, 687, 194, 29, [
        _m('z'), _sub('a'), ' ≤ ', _m('z'), _sub('b')
    ], 22, '#162D19')
    _math(d, 1446, 716, 202, 33, ['⇒ ', *_reward('a'), ' ≥ ', *_reward('b')], 21.0, '#162D19')

    d.text(1109, 851, 128, 30, 'higher reward', 17, TEXT, bold=True)
    gradient_legend = d.shape(1241, 854, 278, 26, '#8CC3F9', '#385D89', 1.05, r=9)
    _gradient(gradient_legend, _palette_at(0), _palette_at(1))
    d.text(1527, 851, 121, 30, 'lower reward', 17, TEXT, bold=True)
