"""Check the handoff artifact without requiring a Windows build on macOS."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location('verify_bundle', ROOT / 'scripts/verify_bundle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowsBundleTests(unittest.TestCase):
    def test_compact_archive_manifest_and_integrity(self):
        archive = ROOT / 'spike_detector_windows_build.zip'
        self.assertTrue(archive.is_file())
        with ZipFile(archive) as zip_file:
            self.assertIsNone(zip_file.testzip())
            names = set(zip_file.namelist())
            manifest = json.loads(zip_file.read('BUNDLE_MANIFEST.json'))
            self.assertEqual(names, set(manifest['sha256']) | {'BUNDLE_MANIFEST.json'})
            self.assertEqual(manifest['app_version'], tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version'])
            self.assertIn('src/spike_detector/utils/validation.py', names)
            self.assertIn('src/spike_detector/utils/widths.py', names)
            self.assertIn('src/spike_detector/utils/batch.py', names)
            self.assertIn('build_windows_uv.bat', names)
            self.assertIn('spike_detector_launcher.py', names)
            self.assertTrue(all(name.startswith(('src/spike_detector/', 'scripts/', 'packaging/pyinstaller/')) or
                                name in {'.python-version','pyproject.toml','requirements.txt','README.md',
                                         'WINDOWS_BUILD.md','build_windows_uv.bat','spike_detector_launcher.py',
                                         'BUNDLE_MANIFEST.json'} for name in names))
            self.assertFalse(any('/data/' in name or name.startswith(('data/','Analysis/','Backup/','.venv/')) or
                                 name.endswith(('.npz','.xlsx','.csv','.exe','.pyc')) for name in names))
            for name, expected in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256(zip_file.read(name)).hexdigest(), expected)

    def test_verifier_rejects_modified_source(self):
        verifier = load_verifier()
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(ROOT/'spike_detector_windows_build.zip') as archive:
                archive.extractall(directory)
            self.assertGreater(verifier.verify_bundle(directory), 10)
            target = Path(directory)/'spike_detector_launcher.py'
            target.write_text(target.read_text() + '\n# changed')
            with self.assertRaisesRegex(ValueError, 'changed or damaged'):
                verifier.verify_bundle(directory)

    def test_spec_and_batch_use_package_and_stop_on_errors(self):
        with ZipFile(ROOT/'spike_detector_windows_build.zip') as archive:
            spec = archive.read('packaging/pyinstaller/spike_detector.spec').decode()
            batch = archive.read('build_windows_uv.bat').decode()
            launcher = archive.read('spike_detector_launcher.py').decode()
        self.assertIn('COLLECT(', spec)
        self.assertIn("ROOT / 'spike_detector_launcher.py'", spec)
        self.assertNotIn('gui_preprocess_V3.3.py', spec)
        self.assertIn('from spike_detector.gui import main', launcher)
        for expected in ('sync --python 3.11 --extra packaging',
                         'scripts\\verify_bundle.py',
                         'packaging\\pyinstaller\\spike_detector.spec',
                         'dist\\spike_detector\\spike_detector.exe',
                         'if errorlevel 1 goto sync_failed',
                         'if errorlevel 1 goto build_failed'):
            self.assertIn(expected, batch)


if __name__ == '__main__':
    unittest.main()
