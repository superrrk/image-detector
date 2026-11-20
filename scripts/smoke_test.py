"""End-to-end smoke test for the real local HTTP endpoint (no Flask required)."""
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PORT = 5097
BASE = f'http://127.0.0.1:{PORT}'


def request(path, *, body=None, content_type=None):
    headers = {'Content-Type': content_type} if content_type else {}
    call = urllib.request.Request(BASE + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(call, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def upload(data, name='face.png'):
    boundary = 'imageforensicslabsmoketest'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            'Content-Type: image/png\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    return request('/api/analyze', body=body, content_type=f'multipart/form-data; boundary={boundary}')


def main():
    proc = subprocess.Popen([sys.executable, '-m', 'server.standalone'], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env={**os.environ, 'PORT': str(PORT)})
    try:
        for _ in range(60):
            try:
                if request('/api/health')[0] == 200:
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        else:
            raise AssertionError('Server did not start')
        status, html = request('/')
        assert status == 200 and b'Image Forensics Lab' in html
        image = io.BytesIO()
        Image.new('RGB', (100, 90), '#567080').save(image, format='PNG')
        status, body = upload(image.getvalue())
        assert status == 200, body[:500]
        payload = json.loads(body)
        assert payload['file']['width'] == 100
        assert payload['metrics']['grayscale_entropy_bits'] >= 0
        assert payload['ela_data_url'].startswith('data:image/png;base64,')
        assert payload['prediction'] is None and payload['model_status'] == 'not_installed'
        status, body = upload(b'invalid', name='bad.png')
        assert status == 415
        status, body = request('/api/analyze', body=b'', content_type='multipart/form-data')
        assert status == 400
        print('PASS: HTML, health, PNG inspection, visualization, absent-model state, invalid file, empty upload')
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


if __name__ == '__main__':
    main()
