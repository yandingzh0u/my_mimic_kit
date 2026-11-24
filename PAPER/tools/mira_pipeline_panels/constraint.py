"""Native PowerPoint panel (c): example of learnable column rescaling."""
from math import exp, log, sqrt


def _example():
    v = ((-.8, .4, -.6), (.3, -.7, .5), (.5, .6, -.4))
    p = [[abs(value) for value in row] for row in v]
    # Show values rounded to three places in the drawing, but perform all
    # operations on unrounded values.
    alpha = [log(value) for value in (1.2, .8, 1.5)]
    q = [exp(value) for value in alpha]
    gram = [[sum(row[i] * row[j] for row in p) for j in range(3)]
            for i in range(3)]
    t = [sum(a * x for a, x in zip(row, q)) / q[i]
         for i, row in enumerate(gram)]
    c = [1.0 / sqrt(value) for value in t]
    w = [[value * c[j] for j, value in enumerate(row)] for row in p]
    return v, p, alpha, q, t, c, w


def draw(d):
    v, p, alpha, q, t, c, w = _example()
    ink = '#071532'
    navy = '#0A2057'
    blue = '#235AAE'
    cell_line = '#365985'
    pale = '#EAF5FD'
    dash = '#B9D1E9'
    stage_fill = ('#F0F7FE', '#DAE9FA')
    blue_cols = ('#B8D9FA', '#E0EFFC', '#DAEBFC')
    green_cols = ('#D6F0CB', '#EBF8E2', '#DAF1D3')
    pink_cols = ('#E8CCED', '#F6E2F8', '#F0D4F4')
    group_styles = (
        (blue_cols, '#5097DF', ('#D9EEFF', '#BBDFFA')),
        (green_cols, '#63A45A', ('#E9F8DB', '#CAEECB')),
        (pink_cols, '#A86ABE', ('#F7E6FA', '#EAC7F1')),
    )

    # Four matrix headers, aligned at a common height.
    stages = [(521, 122), (651, 128), (786, 138), (934, 130)]
    headers = [
        ('Raw', 'weights V'),
        ('Nonnegative', 'weights P = |V|'),
        ('Column-wise', 'scaling'),
        ('Effective', 'weights W'),
    ]
    for (x, width), (first, second) in zip(stages, headers):
        d.shape(x, 461, width, 50, stage_fill, None, r=11)
        d.text(x + 3, 464, width - 6, 22, first,
               size=16.5, color=ink, bold=True)
        d.text(x + 3, 486, width - 6, 20, second,
               size=16.5, color=ink, bold=True)

    def matrix(x, width, values, colors, digits, outline=cell_line):
        y, h = 518, 107
        cell_w, cell_h = width / 3, h / 3
        d.shape(x, y, width, h, pale, None, r=6)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                fill = colors[i][j]
                d.shape(x + j * cell_w + .6, y + i * cell_h + .6,
                        cell_w - 1.2, cell_h - 1.2, fill, None)
                s = f'{value:.{digits}f}'.replace('-', '−')
                d.text(x + j * cell_w + .4, y + i * cell_h + .35,
                       cell_w - .8, cell_h - .7, s,
                       size=13.5 if digits == 3 else 14.5,
                       color=ink, bold=True)
        for ij in (1, 2):
            d.line([(x + ij * cell_w, y + .5), (x + ij * cell_w, y + h - .5)],
                   color='#96AAC4', width=.65)
            d.line([(x + .5, y + ij * cell_h), (x + width - .5, y + ij * cell_h)],
                   color='#96AAC4', width=.65)
        d.shape(x, y, width, h, None, outline, lw=1.3, r=6)

    raw_colors = (
        ('#FAC0BB', '#D7EBFF', '#F8CDC6'),
        ('#E3F0FD', '#F3B5AF', '#C1DBF5'),
        ('#C9DFF6', '#C1DDF8', '#F5D0CB'),
    )
    p_colors = (
        ('#ADD0F5', '#DCEBFC', '#BEDAF7'),
        ('#E3EFFC', '#BDD6F4', '#CEE4FA'),
        ('#C5DDF9', '#BFD8F6', '#E2EEFC'),
    )
    w_colors = (
        ('#AACCF0', '#C7DFF8', '#D3E6FB'),
        ('#E2EEFD', '#BBD6F4', '#D0E4F9'),
        ('#BAD6F6', '#C3DCF7', '#DAEAFB'),
    )
    matrix(525, 117, v, raw_colors, 1)
    matrix(661, 118, p, p_colors, 1)
    matrix(939, 123, w, w_colors, 3)

    # Each highlighted element of c scales exactly one matching column.
    col_x, col_w = [796, 837, 878], 40
    for j, (x, (fills, border, chip)) in enumerate(zip(col_x, group_styles)):
        d.shape(x, 518, col_w, 108, fills[1], None, r=6)
        for i, bg in enumerate(fills):
            d.shape(x + .75, 519 + 35.5 * i, col_w - 1.5, 34.75, bg, None)
            d.text(x + 1, 519 + 35.5 * i, col_w - 2, 34.75,
                   f'{p[i][j]:.1f}', size=14.5, color=ink, bold=True)
        for i in (1, 2):
            yy = 518 + 36 * i
            d.line([(x + 1, yy), (x + col_w - 1, yy)],
                   color=border, width=.58)
        d.shape(x, 518, col_w, 108, None, border, lw=1.4, r=6)
        center_x = x + col_w / 2
        # Chips are slightly wider than columns, spaced to make colors distinct.
        chip_x = 782 + j * 53
        chip_center = chip_x + 24.5
        d.line([(center_x, 626), (center_x, 633), (chip_center, 633),
                (chip_center, 645)], color=border, width=1.35)
        d.shape(chip_x, 645, 49, 30, chip, border, lw=1.4, r=14)
        d.text(chip_x + 2, 646, 45, 28, f'{c[j]:.3f}', size=14,
               color=ink, bold=True)

    for x0, x1 in ((642, 660), (779, 795), (918, 938)):
        d.line([(x0 + .8, 572.5), (x1, 572.5)], color=navy,
               width=1.8, arrow=True)

    # Computation strip and blue upward connector, outside all label regions.
    d.shape(522, 713, 542, 96, ('#F1F9FF', '#E4F1FB'),
            '#CADFEE', lw=1.1, r=10)
    d.line([(1000, 735), (1000, 699.5), (864, 699.5), (864, 678)],
           color=blue, width=2.25, arrow=True)
    d.shape(522, 698, 236, 28, ('#E3F1FF', '#C9E1F8'), None, r=11)
    d.text(535, 698, 216, 28, 'Parameter computation',
           size=18.7, color=ink, bold=True, align='left')

    params = [
        (529, 123, 'learnable α', alpha),
        (668, 118, 'q = exp(α)', q),
        (802, 120, 't', t),
        (938, 119, 'c', c),
    ]
    for x, width, title, values in params:
        d.shape(x, 735, width, 60, ('#F7FCFF', '#EEF7FC'),
                '#6383A8', lw=1.2, r=8)
        if title in ('t', 'c'):
            is_column_factor = title == 'c'
            center = x + (74 if is_column_factor else 76)
            width_f = 26 if is_column_factor else 43
            d.math(center - width_f / 2 - 30, 740, 28, 23,
                   [(title, True, False), (' =', False, False)], 16, ink)
            d.math(center - width_f / 2, 737, width_f, 14,
                   [('1', False, False)] if is_column_factor else [('PᵀPq', True, False)],
                   14, ink)
            d.line([(center - width_f / 2, 751), (center + width_f / 2, 751)], ink, .9)
            if is_column_factor:
                d.radical(center - 8, 754, 16, 10, [('t', True, False)], 13, ink)
            else:
                d.math(center - 12, 751, 24, 13, [('q', True, False)], 13, ink)
        else:
            d.text(x + 2, 739, width - 4, 23, title, size=16,
                   color=ink, bold=(x == params[0][0]),
                   italic=(x != params[0][0]),
                   font='Arial' if x == params[0][0] else 'Liberation Serif')
        vector = '[' + ', '.join(f'{value:.3f}' for value in values) + ']'
        d.text(x + 2, 766, width - 4, 21, vector.replace('-', '−'),
               size=11.7, color=ink, font='Liberation Sans')
    for (a, wa, *_), (b, *_rest) in zip(params, params[1:]):
        d.line([(a + wa, 765), (b, 765)], color=blue,
               width=1.55, arrow=True)

    d.shape(522, 838, 289, 63, ('#E8F2FC', '#D7E9FC'), None, r=12)
    d.text(536, 843, 265, 24, 'Learnable column scaling',
           size=18, color=ink, bold=True, align='left')
    d.text(536, 869, 265, 22, 'monotone and spectrally bounded',
           size=15.8, color=ink, bold=True, align='left')

    for x, width, text in [(819, 109, 'W ≥ 0'), (935, 129, '‖W‖₂ ≤ 1')]:
        d.shape(x, 848, width, 43, ('#F3FBE8', '#E5F5DC'),
                '#9BC47A', lw=1.35, r=9)
        d.check(x + 22, 869.5, 11)
        d.text(x + 35, 850, width - 40, 39, text,
               size=19.5, color=ink, bold=True, font='Liberation Serif')
