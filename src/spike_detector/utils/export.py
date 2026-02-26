import os


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
    figure.savefig(filename, format=save_ext)
    return filename
