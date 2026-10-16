"""Exercise a real killed process and local HTTP transport, without paid API calls."""
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from research.crystal_bulk_annotation_20261013.test_batch import fact
from research.crystal_bulk_annotation_20261013.batch import BatchRunner, STAGE


class RecoveryTest(unittest.TestCase):
    def test_killed_process_preserves_success_and_retries_only_uncommitted_material(self):
        arrived, release = threading.Event(), threading.Event()
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                mid = 'mp-first' if 'mp-first' in payload['messages'][0]['content'] else 'mp-second'
                calls.append(mid)
                if mid == 'mp-second' and calls.count(mid) == 1:
                    arrived.set()
                    release.wait(15)
                event = {'model': 'deepseek-v4-flash', 'choices': [
                    {'delta': {'content': 'A local test crystal description.'}, 'finish_reason': 'stop'}],
                    'usage': {'total_tokens': 10}}
                body = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'facts.jsonl.gz'
                with gzip.open(source, 'wt') as stream:
                    for mid in ('mp-first', 'mp-second'):
                        stream.write(json.dumps(fact(mid, 'train', 0.123456789)) + '\n')
                config = root / 'config.json'
                config.write_text(json.dumps({'base_url': f'http://127.0.0.1:{server.server_port}/v1',
                    'model': 'deepseek-v4-flash', 'thinking': {'type': 'enabled'},
                    'reasoning_effort': 'max', 'stream': True, 'max_tokens': 16384}))
                prompt = root / 'prompt.md'
                prompt.write_text('```text\n{structure_version}\n{core_facts}\n{local_structure_facts}\n{property_labels}\n```')
                env = root / '.env'
                env.write_text('SCIENCEPAL_API_KEY=local-test-only\n')
                run_dir = root / 'run'
                command = [sys.executable, str(STAGE / 'batch.py'), 'run', '--input', str(source),
                    '--run-dir', str(run_dir), '--config', str(config), '--prompt', str(prompt),
                    '--env', str(env), '--workers', '1']
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    self.assertTrue(arrived.wait(10), 'Runner did not reach second HTTP request')
                    runner = BatchRunner([source], run_dir, config, prompt, env, workers=1)
                    self.assertEqual(runner.status()['successes'], 1)
                finally:
                    child.kill()
                    child.communicate(timeout=10)
                    release.set()
                with sqlite3.connect(run_dir / 'annotation.sqlite3') as db:
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM successes').fetchone()[0], 1)
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
                self.assertEqual(calls.count('mp-first'), 1)
                self.assertEqual(calls.count('mp-second'), 2)
                self.assertEqual(runner.status()['successes'], 2)
                self.assertEqual(runner.status()['incomplete_attempts'], 0)
        finally:
            release.set()
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
