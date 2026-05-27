"""Fire-and-forget REST notifications for line-crossing events.

POSTs a small JSON body per crossing in a background thread so a slow or down
endpoint never stalls the capture loop. Failures are logged and ignored.
"""

import logging
import threading

try:
    import requests
except ImportError:
    requests = None


class EventNotifier:
    """POST crossing events as JSON to a URL, asynchronously."""

    def __init__(self, url, timeout=3.0):
        self.url = url
        self.timeout = timeout
        if url and requests is None:
            logging.warning('--api-url set but the "requests" package is missing; API disabled.')

    @property
    def enabled(self):
        return bool(self.url) and requests is not None

    def post(self, payload):
        """Send `payload` (dict) as JSON without blocking the caller."""
        if not self.enabled:
            return
        threading.Thread(target=self._send, args=(payload,), daemon=True).start()

    def _send(self, payload):
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            logging.info(f'[API] {r.status_code} sent {payload}')
        except Exception as e:
            logging.warning(f'[API] FAILED ({e}) {payload}')
