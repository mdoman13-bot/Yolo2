import os
import subprocess
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer

# Proxy server class
class ProxyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"Request received: {self.path}")
        super().do_GET()

def start_proxy_server(port=8080):
    """Start a basic HTTP proxy server."""
    server_address = ("", port)
    with ThreadingTCPServer(server_address, ProxyHandler) as httpd:
        print(f"Proxy server running on port {port}")
        httpd.serve_forever()

def configure_internet_sharing(network_interface, shared_interface):
    """
    Configure macOS Internet Sharing using shell commands.
    :param network_interface: The interface connected to the cruise network (e.g., "en0" for Wi-Fi).
    :param shared_interface: The interface for sharing (e.g., "en1" for Ethernet, "bridge0" for Wi-Fi).
    """
    try:
        # Enable internet sharing
        subprocess.run([
            "sudo", "defaults", "write", "/Library/Preferences/SystemConfiguration/com.apple.nat", "NAT", "-dict-add",
            "Enabled", "1"
        ], check=True)
        
        # Set up NAT and enable Internet Sharing
        subprocess.run([
            "sudo", "/usr/libexec/InternetSharing", "start"
        ], check=True)
        
        print("Internet Sharing configured and started.")
    except subprocess.CalledProcessError as e:
        print(f"Error configuring Internet Sharing: {e}")

def main():
    cruise_network_interface = "en0"  # Replace with your network interface (e.g., Wi-Fi)
    shared_network_interface = "bridge0"  # Replace with the shared interface
    
    # 1. Configure Internet Sharing
    print("Configuring Internet Sharing...")
    configure_internet_sharing(cruise_network_interface, shared_network_interface)
    
    # 2. Start Proxy Server
    print("Starting proxy server...")
    try:
        start_proxy_server(port=8080)
    except KeyboardInterrupt:
        print("\nProxy server stopped.")

if __name__ == "__main__":
    main()