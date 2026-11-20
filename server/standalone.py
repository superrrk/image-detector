"""Dependency-light HTTP demo server: run when Flask is not installed.

Uses only stdlib + Pillow + NumPy. The Flask API remains the primary deployment.
"""
from __future__ import annotations

import io
import json
import os
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from server.core import MAX_BYTES, MAX_PIXELS, ModelUnavailable, Predictor
from server.forensics import inspect_image

STATIC = Path(__file__).resolve().parent / 'static' / 'index.html'
ENGINE = Predictor()


class Handler(BaseHTTPRequestHandler):
    server_version = 'ImageForensicsLab/2.0'

    def reply(self, code: int, payload: dict):
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            data = STATIC.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/api/health':
            self.reply(200, {'status': 'ok', 'model_available': ENGINE.checkpoint.is_file()})
        else:
            self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/api/analyze':
            return self.reply(404, {'error': 'Not found'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0:
                return self.reply(400, {'error': 'Choose an image to analyze.'})
            if size > MAX_BYTES + 16384:
                return self.reply(413, {'error': 'Image exceeds the 8 MB upload limit.'})
            if not self.headers.get('Content-Type', '').startswith('multipart/form-data'):
                return self.reply(400, {'error': 'Multipart upload required.'})
            message = BytesParser(policy=default).parsebytes(
                b'Content-Type: ' + self.headers['Content-Type'].encode('ascii') + b'\r\nMIME-Version: 1.0\r\n\r\n' + self.rfile.read(size))
            parts = [part for part in message.iter_parts() if part.get_param('name', header='content-disposition') == 'image']
            if not parts:
                return self.reply(400, {'error': 'Choose an image to analyze.'})
            raw = parts[0].get_payload(decode=True)
            if not raw:
                return self.reply(400, {'error': 'Choose an image to analyze.'})
            if len(raw) > MAX_BYTES:
                return self.reply(413, {'error': 'Image exceeds the 8 MB upload limit.'})
            with Image.open(io.BytesIO(raw)) as source:
                fmt, has_exif = source.format, bool(source.getexif())
                if fmt not in {'JPEG', 'PNG', 'WEBP'}:
                    return self.reply(415, {'error': 'Only JPEG, PNG and WebP images are supported.'})
                if source.width * source.height > MAX_PIXELS:
                    return self.reply(413, {'error': 'Image dimensions are too large.'})
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
            rotation_text = next((part.get_content() for part in message.iter_parts() if part.get_param('name', header='content-disposition') == 'rotation'), '0')
            try:
                rotation = int(rotation_text)
            except (TypeError, ValueError):
                return self.reply(400, {'error': 'Invalid image rotation.'})
            if rotation not in (0, 90, 180, 270):
                return self.reply(400, {'error': 'Invalid image rotation.'})
            if rotation:
                image = image.rotate(-rotation, expand=True)
            result = inspect_image(image, source_format=fmt, byte_count=len(raw), exif_present=has_exif)
            try:
                result['prediction'] = ENGINE.predict(image)
                result['model_status'] = 'ready'
            except ModelUnavailable:
                result['prediction'] = None
                result['model_status'] = 'not_installed'
            self.reply(200, result)
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
            self.reply(415, {'error': 'That file is not a valid supported image.'})
        except Exception:
            self.log_exception()
            self.reply(500, {'error': 'Analysis failed unexpectedly.'})

    def log_exception(self):
        import traceback
        traceback.print_exc()


def main():
    port = int(os.environ.get('PORT', '5000'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Image Forensics Lab available at http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
