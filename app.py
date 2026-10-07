"""
Standalone Web Application and REST API Server for Iris Flower Classification ANN.
Runs with Python standard library http.server, zero extra dependencies required.
"""

import os
import sys
import json
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import IrisPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "iris_ann_model.json")

# Initialize global predictor
try:
    predictor = IrisPredictor(MODEL_PATH)
except Exception as e:
    print(f"[Warning] Could not initialize predictor immediately: {e}")
    predictor = None


class IrisServerHandler(BaseHTTPRequestHandler):
    """Custom request handler serving web studio files and REST API endpoints."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Root route -> web/index.html
        if path == "/" or path == "/index.html":
            self.serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
            return

        # Static assets in web/
        if path in ["/style.css", "/app.js"]:
            filepath = os.path.join(WEB_DIR, path.lstrip("/"))
            mime = "text/css" if path.endswith(".css") else "application/javascript"
            self.serve_file(filepath, mime)
            return

        # Image assets in web/assets/
        if path.startswith("/assets/"):
            filepath = os.path.join(WEB_DIR, path.lstrip("/"))
            mime, _ = mimetypes.guess_type(filepath)
            self.serve_file(filepath, mime or "image/jpeg")
            return

        # Generated visualization images in results/
        if path.startswith("/results/"):
            filename = os.path.basename(path)
            filepath = os.path.join(RESULTS_DIR, filename)
            mime, _ = mimetypes.guess_type(filepath)
            self.serve_file(filepath, mime or "image/png")
            return

        # API: Model Info & Metrics
        if path == "/api/model_info":
            if predictor and os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                self.send_json_response(meta)
            else:
                self.send_json_response({"error": "Model not loaded. Run train.py first."}, status=404)
            return

        # Default 404
        self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            global predictor
            if predictor is None:
                try:
                    predictor = IrisPredictor(MODEL_PATH)
                except Exception as e:
                    self.send_json_response({"error": f"Model failed to load: {e}"}, status=500)
                    return

            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len).decode("utf-8")
                payload = json.loads(post_body)

                sl = float(payload.get("sepal_length", 5.1))
                sw = float(payload.get("sepal_width", 3.5))
                pl = float(payload.get("petal_length", 1.4))
                pw = float(payload.get("petal_width", 0.2))

                result = predictor.predict(sl, sw, pl, pw)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=400)
            return

        self.send_error(404, "Endpoint Not Found")

    def serve_file(self, filepath: str, mime_type: str):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {os.path.basename(filepath)}")
            return
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Error: {e}")

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress routine GET logs for clean console
        return


def start_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, IrisServerHandler)
        print("=" * 65)
        print("      IRIS ARTIFICIAL NEURAL NETWORK (ANN) WEB STUDIO          ")
        print("=" * 65)
        print(f"  * Server Status : ONLINE")
        print(f"  * Access Web App: http://127.0.0.1:{port}")
        print(f"  * REST API Docs : POST http://127.0.0.1:{port}/api/predict")
        print("=" * 65)
        print("Press Ctrl+C to terminate the web server.\n")
        httpd.serve_forever()
    except OSError:
        # Fallback to port 5001 if 5000 is busy
        print(f"Port {port} in use, trying {port+1}...")
        start_server(port + 1)
    except KeyboardInterrupt:
        print("\nShutting down web server. Goodbye!")


if __name__ == "__main__":
    start_server(5000)
