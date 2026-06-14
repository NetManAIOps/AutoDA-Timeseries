import socks
import socket

from utils import GlobalConfig


class ProxyContext:
    def __init__(self, global_config:GlobalConfig):
        try:
            proxy_host, proxy_port = global_config.args.socks_proxy.split(":")
            self.proxy_host = proxy_host
            self.proxy_port = int(proxy_port)
        except:
            self.proxy_port = None
            self.proxy_host = None
        self._original_socket = socket.socket
        self._orig_getaddrinfo = socket.getaddrinfo

    def __enter__(self):
        if self.proxy_host and self.proxy_port:
            # Set the SOCKS proxy for this context
            socks.set_default_proxy(socks.SOCKS5, self.proxy_host, self.proxy_port)
            socket.socket = socks.socksocket

            def getaddrinfo_ipv4(*args, **kwargs):
                responses = self._orig_getaddrinfo(*args, **kwargs)
                return [response for response in responses if response[0] == socket.AF_INET]

            socket.getaddrinfo = getaddrinfo_ipv4

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original socket
        socket.socket = self._original_socket
        socket.getaddrinfo = self._orig_getaddrinfo