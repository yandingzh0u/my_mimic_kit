"""Panel (b): editable group-normalization illustration for the MIRA diagram."""

import math


INK = '#101F3B'
NAVY = '#102757'
BLUE = '#1764BD'
ORANGE = '#E97919'
RULE = '#AAC5E0'
GRAPH_BG = '#F9FCFF'


def draw(d):
    """Paint panel (b) inside the corresponding frame on the overview slide."""
    def strip(y, number, title, symbol):
        d.shape(22, y, 464, 31, ('#CFE3F8', '#BBD7F1'), r=8)
        d.oval(34, y + 4.5, 22, 22, '#E4F1FC', NAVY, 1.35)
        d.text(34, y + 4.3, 22, 22, str(number), 14, NAVY, True)
        if symbol == 'σ':
            d.text(69, y + 1, 251, 28, title, 18, INK, True, 'left')
            d.text(319, y + 1, 17, 27, 'σ', 20, INK, False, font='Times New Roman', italic=True)
            d.text(332, y + 14, 10, 14, 'g', 13, INK, False, font='Times New Roman', italic=True)
        else:
            d.text(69, y + 1, 216, 28, title, 18, INK, True, 'left')
            d.text(284, y + 1, 39, 27, '√d', 20, INK, False, font='Times New Roman', italic=True)
            d.text(312, y + 14, 10, 14, 'g', 13, INK, False, font='Times New Roman', italic=True)

    strip(458, 1, 'Normalize by historical scale', 'σ')
    for x, icon_name, color, colors, border, label, unit in (
        (28, 'pin', BLUE, ('#F6FCF0', '#EDF6E3'), '#CFDFC4', 'position', 'm'),
        (111, 'rotation', BLUE, ('#F3F9FE', '#E6F1FA'), '#CCDEEC', 'rotation', 'rad'),
        (192, 'gauge', '#144995', ('#FFF8EF', '#FAEBDD'), '#F1DAC1', 'velocity', 'rad · s⁻¹'),
    ):
        d.shape(x, 500, 78, 83, colors, border, 1, 8)
        d.icon(icon_name, x + 39, 520, 28, color)
        d.text(x + 1, 534, 76, 20, label, 14, INK, True)
        d.text(x + 1, 556, 76, 16, unit, 13, INK)

    d.poly([(275, 536), (289, 536), (289, 530), (303, 542), (289, 554), (289, 548), (275, 548)], '#8494A6')
    d.shape(309, 500, 172, 83, ('#FAFCFF', '#F1F6FB'), '#C6D8E9', 1, 9)
    for y, width, gradient, border in (
        (511, 91, ('#A5D671', '#84BB51'), '#679B47'),
        (535, 61, ('#B6DDF8', '#8CC3EE'), '#498FCE'),
        (559, 127, ('#FFC77F', '#F5A447'), '#D78437'),
    ):
        d.shape(320, y, 150, 14, '#E2EAF0', r=4)
        d.shape(320, y, width, 14, gradient, border, 0.8, 4)

    d.shape(22, 592, 464, 34, ('#F6FCF2', '#E8F5E2'), r=8)
    d.check(47, 609, 10)
    d.text(66, 594, 411, 28, 'Reduces unit and scale differences across error types.', 14, '#183C24', True, 'left')
    d.line([(22, 634), (486, 634)], RULE, 1.25)

    strip(645, 2, 'Normalize by group size', '√d')

    # Equal per-component, scale-normalized errors produce a sqrt(d_g) L2
    # norm before dimension normalization and a constant L2 norm after it.
    # Both lines are editable PPT paths, with no measured data implied.
    x_axis, y_axis = 118, 790
    d.line([(x_axis, y_axis), (365, y_axis)], NAVY, 1.8, True)
    d.line([(x_axis, y_axis), (x_axis, 688)], NAVY, 1.8, True)
    d.text(25, 690, 79, 16, 'Group Input', 14, INK, True)
    d.text(25, 708, 79, 16, 'L2 Norm', 14, INK, True)
    d.text(174, 796, 136, 18, 'Group Dimension', 14, INK, True, 'left')
    d.text(298, 796, 12, 18, 'd', 17, INK, False, font='Times New Roman', italic=True)
    d.text(307, 807, 9, 10, 'g', 10, INK, False, font='Times New Roman', italic=True)
    left, right = 121, 348
    dim_min, dim_max = 1.0, 12.0
    norm_at_one, norm_px = 764.0, 20.5
    dims = [dim_min + (dim_max - dim_min) * k / 32 for k in range(33)]

    def points(divisor):
        return [
            (left + (right - left) * (n - dim_min) / (dim_max - dim_min),
             norm_at_one - norm_px * (math.sqrt(n) / divisor(n) - 1))
            for n in dims
        ]

    d.line(points(lambda n: 1.0), ORANGE, 2.9)
    d.line(points(math.sqrt), BLUE, 2.75)

    d.shape(365, 711, 120, 65, GRAPH_BG, '#BDD1E4', 1, 7)
    for y, color, text in ((730, ORANGE, 'Without'), (757, BLUE, 'With')):
        d.line([(372, y), (399, y)], color, 2.7)
        d.text(404, y - 10, 51, 18, text, 12, INK, True, 'left')
        d.text(455, y - 11, 21, 19, '√d', 14, INK, False, font='Times New Roman', italic=True)
        d.text(474, y, 7, 11, 'g', 9, INK, False, font='Times New Roman', italic=True)

    d.shape(50, 830, 409, 75, ('#ECF5FE', '#CDE2F7'), r=11)
    d.text(57, 842, 144, 21, 'Group', 17, INK, True)
    d.text(57, 865, 144, 21, 'Normalization', 17, INK, True)
    d.shape(207, 831, 251, 73, ('#FCFEFF', '#E9F4FD'), '#91AFD1', 1, 10)
    # Typeset a native editable fraction, including explicit subscripts and
    # superscripts in their own boxes rather than depending on fallback glyphs.
    serif = 'Times New Roman'
    d.text(230, 857, 16, 28, 'z', 24, INK, False, font=serif, italic=True)
    d.text(242, 872, 8, 12, 't', 12, INK, False, font=serif, italic=True)
    d.text(244, 849, 23, 18, '(g)', 15, INK, False, font=serif, italic=True)
    d.text(272, 856, 19, 26, '=', 22, INK, False, font=serif)
    d.text(355, 838, 15, 26, 'e', 23, INK, False, font=serif, italic=True)
    d.text(368, 853, 8, 11, 't', 12, INK, False, font=serif, italic=True)
    d.text(369, 834, 24, 17, '(g)', 15, INK, False, font=serif, italic=True)
    d.line([(299, 868), (441, 868)], INK, 1.1)
    d.text(300, 871, 52, 24, 'max(', 20, INK, False, 'left', font=serif)
    d.text(343, 871, 17, 24, 'σ', 20, INK, False, font=serif, italic=True)
    d.text(355, 886, 9, 13, 'g', 12, INK, False, font=serif, italic=True)
    d.text(364, 871, 15, 24, ',', 20, INK, False, 'left', font=serif)
    d.text(375, 871, 11, 24, 'ε', 20, INK, False, font=serif, italic=True)
    d.text(387, 871, 11, 24, ')', 20, INK, False, font=serif)
    d.text(399, 871, 34, 24, '√d', 20, INK, False, font=serif, italic=True)
    d.line([(415, 876), (439, 876)], INK, 0.7)
    d.text(429, 886, 9, 13, 'g', 12, INK, False, font=serif, italic=True)
