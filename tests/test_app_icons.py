"""Check native icon formats, runtime artwork and platform packaging paths."""
import os
import base64
import xml.etree.ElementTree as ET
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spike-detector-test-mpl')
from pathlib import Path
import tomllib
import types
import unittest
from unittest.mock import patch
from PIL import Image
from PyQt6 import QtWidgets
from spike_detector.gui import MainWindow, _application_icon

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'src'/'spike_detector'/'assets'


class IconTests(unittest.TestCase):
    def test_native_icons_have_small_and_large_representations(self):
        with Image.open(ASSETS/'spike_detector.png') as source:
            self.assertEqual(source.size, (190, 190))
            self.assertEqual(source.format, 'PNG')
        with Image.open(ASSETS/'spike_detector.ico') as icon:
            self.assertEqual(icon.format, 'ICO')
            self.assertTrue({(16, 16), (32, 32), (48, 48), (256, 256)} <= icon.info['sizes'])
            icon.load()
        with Image.open(ASSETS/'spike_detector.icns') as icon:
            self.assertEqual(icon.format, 'ICNS')
            self.assertEqual(icon.size, (1024, 1024))
            icon.load()

    def test_runtime_icon_uses_supplied_artwork_and_main_window_has_it(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with patch('spike_detector.gui.sys.platform', 'win32'):
            icon = _application_icon()
        self.assertFalse(icon.isNull())
        image = icon.pixmap(190, 190).toImage()
        with Image.open(ASSETS/'spike_detector.png') as source:
            pixels = source.convert('RGB')
            for x, y in ((27, 25), (48, 25), (95, 130), (0, 0)):
                self.assertEqual(image.pixelColor(x, y).getRgb()[:3], pixels.getpixel((x, y)))
        window = MainWindow()
        try:
            self.assertFalse(window.windowIcon().isNull())
        finally:
            window.close()

    def test_macos_native_and_runtime_icons_have_transparent_outer_corners(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        svg = ET.fromstring((ASSETS/'spike_detector_macos.svg').read_text())
        image_node = svg.find('{http://www.w3.org/2000/svg}image')
        embedded = image_node.attrib['{http://www.w3.org/1999/xlink}href'].split(',', 1)[1]
        self.assertEqual(base64.b64decode(embedded), (ASSETS/'spike_detector.png').read_bytes())
        with Image.open(ASSETS/'spike_detector_macos.png') as png:
            self.assertEqual(png.width, png.height)
            self.assertIn('A', png.getbands())
            for point in ((0, 0), (png.width-1, 0), (0, png.height-1), (png.width-1, png.height-1)):
                self.assertEqual(png.getpixel(point)[3], 0)
            self.assertEqual(png.getpixel((png.width//2, png.height//2))[3], 255)
        with Image.open(ASSETS/'spike_detector.icns') as icns:
            image = icns.convert('RGBA')
            self.assertEqual(image.getpixel((0, 0))[3], 0)
            self.assertEqual(image.getpixel((512, 512))[3], 255)
        with patch('spike_detector.gui.sys.platform', 'darwin'):
            qt_image = _application_icon().pixmap(128, 128).toImage()
        self.assertEqual(qt_image.pixelColor(0, 0).alpha(), 0)
        self.assertEqual(qt_image.pixelColor(64, 64).alpha(), 255)

    def test_installed_package_declares_icon_assets(self):
        project = tomllib.loads((ROOT/'pyproject.toml').read_text())
        data = project['tool']['setuptools']['package-data']['spike_detector']
        self.assertTrue({'assets/*.png', 'assets/*.ico', 'assets/*.icns'} <= set(data))

    def test_windows_and_macos_specs_embed_correct_native_icon_and_runtime_assets(self):
        calls = []
        def target(name):
            def make(*args, **kwargs):
                result = types.SimpleNamespace(kind=name, args=args, kwargs=kwargs, pure=[] , scripts=[], binaries=[], datas=[])
                calls.append(result)
                return result
            return make
        hooks = types.ModuleType('PyInstaller.utils.hooks')
        hooks.collect_data_files = lambda name: []
        code = (ROOT/'packaging/pyinstaller/spike_detector.spec').read_text()
        for platform in ('win32', 'darwin'):
            with self.subTest(platform=platform):
                calls.clear()
                namespace = {'SPEC': str(ROOT/'packaging/pyinstaller/spike_detector.spec')}
                namespace.update({name: target(name) for name in ('Analysis', 'PYZ', 'EXE', 'COLLECT', 'BUNDLE')})
                with patch('sys.platform', platform), patch.dict('sys.modules', {'PyInstaller.utils.hooks': hooks}):
                    exec(compile(code, namespace['SPEC'], 'exec'), namespace)
                analysis = next(c for c in calls if c.kind == 'Analysis')
                self.assertIn((str(ASSETS), 'spike_detector/assets'), analysis.kwargs['datas'])
                exe = next(c for c in calls if c.kind == 'EXE')
                if platform == 'win32':
                    self.assertEqual(exe.kwargs['icon'], str(ASSETS/'spike_detector.ico'))
                    self.assertFalse(any(c.kind == 'BUNDLE' for c in calls))
                else:
                    bundle = next(c for c in calls if c.kind == 'BUNDLE')
                    self.assertEqual(bundle.kwargs['icon'], str(ASSETS/'spike_detector.icns'))
                    self.assertEqual(bundle.kwargs['name'], 'Spike Detector.app')
                    self.assertEqual(bundle.kwargs['version'], tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version'])
                    self.assertTrue(bundle.kwargs['info_plist']['NSHighResolutionCapable'])


if __name__ == '__main__':
    unittest.main()
