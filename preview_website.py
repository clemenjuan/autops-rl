#!/usr/bin/env python3
"""
Simple HTTP server to preview the website locally
Usage: python preview_website.py
Then open http://localhost:8000 in your browser
"""

import http.server
import socketserver
import webbrowser
import os
import socket

def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    port = find_free_port(8000)
    if port is None:
        print("Error: Could not find a free port. Please close other servers or try manually:")
        print("  python -m http.server 8001")
        return
    
    try:
        with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
            print(f"Server running at http://localhost:{port}/")
            print("Press Ctrl+C to stop the server")
            print(f"Opening browser at http://localhost:{port}/index.html")
            try:
                webbrowser.open(f'http://localhost:{port}/index.html')
            except:
                print(f"Could not open browser automatically. Please visit: http://localhost:{port}/index.html")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
    except OSError as e:
        print(f"Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Check if another server is running: lsof -i :8000 (Linux/Mac) or netstat -ano | findstr :8000 (Windows)")
        print("2. Kill the process using the port or use a different port:")
        print("   python -m http.server 8001")

if __name__ == "__main__":
    main()

