#!/usr/bin/env python3
"""
代理服务器 - 解决CORS问题
前端和API都通过8080端口访问
"""

import http.server
import socketserver
import urllib.request
import urllib.parse
import json
from pathlib import Path
import os

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/'):
            # 代理API请求到8000端口
            self.proxy_to_api('GET')
        else:
            # 提供前端静态文件
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/'):
            # 代理API请求到8000端口
            self.proxy_to_api('POST')
        else:
            self.send_error(404)

    def proxy_to_api(self, method):
        try:
            # 移除/api前缀
            api_path = self.path[4:]  # 去掉 /api
            api_url = f'http://localhost:8000{api_path}'

            if method == 'POST':
                # 读取POST数据
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)

                req = urllib.request.Request(api_url, data=post_data, method=method)
                req.add_header('Content-Type', 'application/json')
            else:
                req = urllib.request.Request(api_url, method=method)

            # 发送请求到API服务器
            with urllib.request.urlopen(req) as response:
                # 读取响应数据
                response_data = response.read()

                # 设置响应头
                self.send_response(response.getcode())
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(response_data)))
                self.end_headers()

                # 返回响应数据
                self.wfile.write(response_data)

        except Exception as e:
            print(f"代理错误: {e}")
            # 发送JSON格式的错误响应
            error_response = json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(error_response)))
            self.end_headers()
            self.wfile.write(error_response)

def start_proxy_server(port=8080):
    """启动代理服务器"""

    frontend_dir = Path("web_frontend")
    if not frontend_dir.exists():
        print("❌ web_frontend 目录不存在")
        return False

    # 切换到前端目录
    os.chdir(frontend_dir)

    print(f"🌐 启动代理服务器在端口 {port}")
    print(f"📱 前端访问: http://localhost:{port}")
    print(f"🔗 API代理: http://localhost:{port}/api/")
    print("💡 按 Ctrl+C 停止服务")

    try:
        with socketserver.TCPServer(("0.0.0.0", port), ProxyHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  代理服务已停止")
        return True
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

if __name__ == "__main__":
    start_proxy_server()