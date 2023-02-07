import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import urllib


class ExampleHandler(BaseHTTPRequestHandler):
    def __init__(self, *args):
        super(ExampleHandler, self).__init__(*args)

    def setup_headers(self):
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*8')
        self.send_header('Access-Control-Allow-Methods', 'OPTIONS,POST,GET')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(url.query)
        data = b''
        if url.path == '/sum':
            content_len = int(self.headers.get('Content-Length'))
            body = self.rfile.read(content_len)
            data = json.loads(body)
            data = {'result': data.get('a', 0) + data.get('b', 0)}
            data = json.dumps(data).encode()
        self.send_response(200)
        self.setup_headers()
        self.end_headers()
        self.wfile.write(data)


class HttpHelper:
    def __init__(self):
        pass

    def run(self):
        server = HTTPServer(('', 8197), lambda *args: ExampleHandler(*args))
        while True:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                break
        server.server_close()


def main():
    server = HttpHelper()
    server.run()


if __name__ == "__main__":
    main()

