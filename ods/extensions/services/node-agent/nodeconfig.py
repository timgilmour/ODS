import os
import socket

NODE_AGENT_KEY = os.environ.get("NODE_AGENT_KEY", "")
NODE_NAME = os.environ.get("NODE_NAME", socket.gethostname())
GPU_BACKEND = os.environ.get("GPU_BACKEND", "nvidia").lower()
NODE_SERVING_PROBE_URL = os.environ.get("NODE_SERVING_PROBE_URL", "")
NODE_SERVING_CONTAINER = os.environ.get("NODE_SERVING_CONTAINER", "")
NODE_AGENT_PORT = int(os.environ.get("NODE_AGENT_PORT", "7720"))
GPU_CACHE_TTL_SECONDS = float(os.environ.get("NODE_GPU_CACHE_TTL", "2.0"))
