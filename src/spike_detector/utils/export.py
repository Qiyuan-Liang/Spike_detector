import os
import matplotlib as mpl
from matplotlib.text import Text


def save_figure_with_dialog(get_save_filename, figure, default_name='figure.svg'):
    formats = ['SVG (*.svg)', 'PNG (*.png)', 'JPG (*.jpg)', 'PDF (*.pdf)']
    filename, _ = get_save_filename(
        'Save Figure',
        os.path.expanduser(f'~/{default_name}'),
        ';;'.join(formats),
    )
    if not filename:
        return None

    _, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip('.')
    if ext not in ('svg', 'pdf', 'png', 'jpg', 'jpeg'):
        filename = filename + '.svg'
        ext = 'svg'
    save_ext = 'jpg' if ext == 'jpeg' else ext
    rc = {'svg.fonttype': 'none', 'pdf.fonttype': 42} if save_ext in ('svg', 'pdf') else {}
    for artist in figure.findobj(match=Text):
        artist.set_fontfamily('Arial')
    with mpl.rc_context(rc):
        figure.savefig(filename, format=save_ext)
    return filename
