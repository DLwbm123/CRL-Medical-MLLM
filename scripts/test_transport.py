"""Exercise native resume after a failed request and reject an oversized partial."""
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from p13_download import transfer_file


def main():
    data = b"synthetic-resource" * 100
    ranges = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            ranges.append(self.headers.get("Range"))
            if len(ranges) == 1:
                self.send_response(503); self.send_header("Content-Length", "0"); self.end_headers(); return
            offset = int(self.headers["Range"].split("=")[1].split("-")[0])
            self.send_response(206); self.send_header("Content-Length", str(len(data) - offset))
            self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
            self.end_headers(); self.wfile.write(data[offset:])
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as temporary:
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        proxy = f"http://127.0.0.1:{server.server_port}"
        path = Path(temporary) / "partial"; path.write_bytes(data[:100])
        transfer_file("http://synthetic.invalid/resource", path, len(data), proxy)
        assert path.read_bytes() == data and ranges == ["bytes=100-", "bytes=100-"]
        try: transfer_file("http://synthetic.invalid/resource", path, len(data) - 1, proxy)
        except ValueError: pass
        else: raise AssertionError("Oversized retained partial accepted")
        server.shutdown(); thread.join(timeout=5)
    print("Native proxy resume, retained prefix and size rejection passed")


if __name__ == "__main__":
    main()
