"""Build native icons from unchanged artwork and a deterministic macOS tile.

Run with Python, Pillow and PyQt6: python scripts/create_app_icons.py
Windows uses the original PNG; macOS uses a rounded SVG frame around the same
embedded artwork. No waveform or spike-tick detail is redrawn.
"""
import base64
from pathlib import Path
from PIL import Image
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'src' / 'spike_detector' / 'assets'


def create_icons():
    source_path = ASSETS / 'spike_detector.png'
    with Image.open(source_path) as source:
        if source.width != source.height:
            raise ValueError('Program artwork must be square to preserve its proportions.')
        master = source.convert('RGBA').resize((1024, 1024), Image.Resampling.LANCZOS)
        master.save(ASSETS / 'spike_detector.ico', format='ICO',
                    sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])

    # A square decal inset within the rounded white tile fits entirely inside its
    # corners. Keep the original artwork intact, without clipping or retouching.
    # Qt's vector renderer supplies smooth edges and a truly transparent canvas.
    decal = base64.b64encode(source_path.read_bytes()).decode('ascii')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
    width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect x="96" y="96" width="832" height="832" rx="180" ry="180" fill="white"/>
  <image x="160" y="160" width="704" height="704" xlink:href="data:image/png;base64,{decal}"/>
</svg>
'''
    (ASSETS / 'spike_detector_macos.svg').write_text(svg, encoding='utf-8')
    renderer = QSvgRenderer(svg.encode('utf-8'))
    if not renderer.isValid():
        raise ValueError('Invalid macOS icon SVG.')
    canvas = QImage(1024, 1024, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(0)
    painter = QPainter(canvas)
    renderer.render(painter)
    painter.end()
    mac_png = ASSETS / 'spike_detector_macos.png'
    if not canvas.save(str(mac_png), 'PNG'):
        raise OSError('Failed to save rounded macOS icon.')
    with Image.open(mac_png) as mac_source:
        mac_source.save(ASSETS / 'spike_detector.icns', format='ICNS')
    print(f'Created Windows ICO and rounded macOS SVG/PNG/ICNS in {ASSETS}')


if __name__ == '__main__':
    create_icons()
