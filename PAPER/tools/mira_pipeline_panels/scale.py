"""Panel (b): two stages of group normalization, drawn in native PPT."""

import math


INK = '#101F3B'
NAVY = '#102757'
BLUE = '#1B57CF'
ORANGE = '#EB7F15'
RULE = '#AAC5E0'
MATH = 'Liberation Serif'
RMS_A, RMS_B = .62, 1.0


def _example_norms(size):
    """Orange and blue use different dimensionless per-component RMS values."""
    return RMS_A * math.sqrt(size), RMS_B


def draw(d):
    """Paint panel (b) inside the corresponding frame on the overview slide."""
    group_dimension = [('d', True, False), ('g', True, True)]

    def strip(y, number, title):
        d.shape(22, y, 464, 28, ('#D8E8FC', '#B2CEF5'), r=7)
        d.oval(34, y + 3, 22, 22, '#DFEBFD', NAVY, 1.35)
        d.text(34, y + 3, 22, 22, str(number), 14, NAVY, True)
        d.text(68, y + 1, 250, 26, title, 17.7, INK, True, 'left')
        if number == 1:
            d.math(316, y, 39, 26, [('σ', True, False), ('g', True, True)], 20)
        else:
            d.radical(277, y + 5, 28, 19, group_dimension, 18)

    def benefit(y, label):
        d.shape(22, y, 464, 25, ('#F3FAEC', '#E5F3DA'), r=8)
        d.check(44, y + 12.5, 8.6)
        d.text(61, y, 417, 25, label, 13.5, '#183C24', True, 'left')

    strip(458, 1, 'Normalize by historical scale')
    card_y, card_h = 493, 112
    for x, w, title in (
        (28, 203, 'Physical errors'),
        (289, 192, 'Dimensionless errors'),
    ):
        d.shape(x, card_y, w, card_h, ('#F4F8FF', '#EBF3FC'), '#91ACCD', 1.15, 8)
        d.shape(x + .65, card_y + .65, w - 1.3, 34, ('#DFECFC', '#D5E5FC'), r=7)
        d.shape(x + .65, card_y + 24, w - 1.3, 10.7, '#D5E5FC')
        d.text(x + 2, 494, w - 4, 32, title, 16.5, INK, True)

    scales = (
        ('pin', '#278B62', 'position', 'p', 'm',
         ('#F0FAE9', '#E4F3DB'), '#BDDBAD', ('#8BCB5F', '#65AD47'), '#588F42', .60),
        ('rotation', '#155DC6', 'rotation', 'r', 'rad',
         ('#EDF6FF', '#D9EAFD'), '#AECCEA', ('#ADDAFF', '#71B4F6'), '#438CCF', .39),
        ('gauge', '#184687', 'velocity', 'v', 'rad · s⁻¹',
         ('#FFF8EE', '#FAEBD6'), '#F0D6B5', ('#FFC66D', '#F6A139'), '#D88831', .83),
    )
    for i, (icon, color, label, group, units, card, border, bar, bar_border,
            bar_fraction) in enumerate(scales):
        y = 532 + 22 * i
        d.shape(35, y, 188, 18, card, border, .8, 4)
        d.icon(icon, 51, y + 9, 16, color)
        d.text(65, y, 59, 18, label, 14, INK, True, 'left')
        d.text(128, y - .5, 9, 18, 'e', 15.5, INK, italic=True, font=MATH)
        d.text(135, y - 1.4, 18, 11, f'({group})', 9.6, INK, italic=True, font=MATH)
        d.line([(161, y + 1), (161, y + 17)], border, .7)
        d.text(164, y, 55, 18, units, 12.2, INK)
        d.shape(300, y + 2, 169, 14, '#D8E1EC', r=4)
        d.shape(300, y + 2, 169 * bar_fraction, 14, bar, bar_border, .9, 4)

    d.math(236, 530, 50, 24,
           [('÷ ', False, False), ('σ', True, False), ('g', True, True)], 19)
    d.poly([(241, 562), (264, 562), (264, 557), (279, 568),
            (264, 579), (264, 574), (241, 574)], '#8292A1')
    benefit(613, 'Puts different error types on dimensionless scales.')
    d.line([(22, 645), (486, 645)], RULE, 1.2)

    strip(652, 2, 'Normalize by group size')
    x_axis, y_axis = 104, 782
    d.line([(x_axis, y_axis), (338, y_axis)], NAVY, 1.8, arrow=True)
    d.line([(x_axis, y_axis), (x_axis, 689)], NAVY, 1.8, arrow=True)
    d.text(23, 696, 74, 16, 'Group input', 13.5, INK, True)
    d.text(23, 712, 74, 16, 'L₂ norm', 13.5, INK, True)
    d.text(147, 785, 118, 19, 'Group dimension', 14, INK, True, 'left')
    d.math(260, 784, 24, 20, group_dimension, 17)

    # A has lower per-component RMS than B. For paired inputs from the
    # same group the orange norm would never be smaller than the blue norm.
    dims = [1. + 11. * k / 40 for k in range(41)]
    norms = [_example_norms(n) for n in dims]
    for column, color in ((1, BLUE), (0, ORANGE)):
        points = [(109 + 217 * (n - 1.) / 11., y_axis - 39 * v[column])
                  for n, v in zip(dims, norms)]
        curve = d.line(points, color, 2.9)
        curve.name = ('Group B — with dimension scaling' if column
                      else 'Group A — without dimension scaling')

    d.shape(345, 712, 136, 58, ('#FAFDFF', '#EEF5FC'), '#A7BFD9', 1, 7)
    for y, color, text in (
        (727, ORANGE, 'Group A before'), (754, BLUE, 'Group B after'),
    ):
        d.line([(353, y), (373, y)], color, 2.7)
        d.text(379, y - 9, 97, 18, text, 12.8, INK, True, 'left')

    benefit(825, 'Makes group norms comparable across different sizes.')
    d.shape(52, 859, 409, 48, ('#ECF5FE', '#CFDEF4'), r=9)
    d.text(58, 862, 127, 19, 'Group', 15.2, INK, True)
    d.text(58, 880, 127, 20, 'Normalization', 15.2, INK, True)
    d.shape(191, 859, 270, 48, ('#FCFEFF', '#EDF5FD'), '#91AFD1', 1, 8)

    def symbol_with_indices(x, y, symbol, size):
        d.text(x, y, 13, 23, symbol, size, INK, italic=True, font=MATH)
        d.text(x + 9, y + 12, 8, 11, 't', 10.3, INK, italic=True, font=MATH)
        d.text(x + 12, y - 2, 19, 14, '(g)', 12.4, INK, italic=True, font=MATH)

    symbol_with_indices(211, 871, 'z', 19.5)
    d.text(244, 872, 24, 21, '=', 19.5, INK, font=MATH)
    symbol_with_indices(343, 859, 'e', 19.5)
    d.line([(277, 882), (440, 882)], INK, 1)
    d.math(306, 882, 75, 22,
           [('max(', False, False), ('σ', True, False), ('g', True, True),
            (', ε)', False, False)], 17.7)
    d.radical(383, 885, 27, 17, group_dimension, 17.7)
