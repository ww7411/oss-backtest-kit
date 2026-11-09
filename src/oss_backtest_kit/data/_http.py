"""A deliberately strict HTTPS GET client for public market-data hosts.

Safety properties:

* one allowed host, https only, default port only
* no proxy settings taken from the environment
* redirects are never followed (a redirect is an error)
* the response body is size-bounded and the whole request has a deadline
* the opener is injectable so tests can run fully offline
"""

import time
import urllib.error
import urllib.parse
import urllib.request
import ssl

__all__ = ["DataError", "RefusedError", "TooLargeError", "TransientError", "HttpClient"]


class DataError(Exception):
    """Base class for download problems."""


class RefusedError(DataError):
    """The request or response broke a safety rule (host, scheme, redirect)."""


class TooLargeError(DataError):
    """The response body exceeded the size limit."""


class TransientError(DataError):
    """A retryable failure (network error, timeout, server error)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError for the 3xx response


def default_opener():
    """Opener without environment proxies and without redirect following."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect,
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    ).open


class HttpClient:
    """GET client bound to a single host.

    ``opener(request, timeout=...)`` must return a file-like response with
    ``read``/``read1``, ``geturl()`` and ``headers``. Defaults to a hardened
    urllib opener.
    """

    def __init__(self, allowed_host, opener=None, max_bytes=64 * 1024 * 1024,
                 deadline_s=300.0, socket_timeout_s=30.0, retries=4,
                 backoff_s=2.0, sleep=time.sleep, clock=time.monotonic,
                 user_agent="oss-backtest-kit"):
        self.allowed_host = allowed_host
        self.opener = opener or default_opener()
        self.max_bytes = max_bytes
        self.deadline_s = deadline_s
        self.socket_timeout_s = socket_timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.clock = clock
        self.user_agent = user_agent

    def _check_url(self, url):
        u = urllib.parse.urlsplit(url)
        if u.scheme != "https" or u.hostname != self.allowed_host or u.port not in (None, 443):
            raise RefusedError("refused url: %s" % url)
        if "@" in u.netloc:
            raise RefusedError("refused url with embedded userinfo: %s" % url)

    def _once(self, url):
        self._check_url(url)
        start = self.clock()
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            resp = self.opener(req, timeout=self.socket_timeout_s)
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code == 404:
                return None
            if 300 <= code < 400:
                raise RefusedError("redirect refused (%d): %s" % (code, url))
            if code == 429 or code >= 500:
                raise TransientError("http %d: %s" % (code, url))
            raise DataError("http %d: %s" % (code, url))
        except (urllib.error.URLError, OSError) as exc:
            raise TransientError("network error %r: %s" % (exc, url))
        try:
            status = getattr(resp, "status", 200)
            if 300 <= status < 400:
                raise RefusedError("redirect refused (%d): %s" % (status, url))
            if urllib.parse.urlsplit(resp.geturl()).hostname != self.allowed_host:
                raise RefusedError("unexpected final url: %s" % resp.geturl())
            parts, total = [], 0
            read = getattr(resp, "read1", None) or resp.read
            while True:
                chunk = read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.max_bytes:
                    raise TooLargeError("response larger than %d bytes: %s" % (self.max_bytes, url))
                if self.clock() - start > self.deadline_s:
                    raise TransientError("deadline exceeded: %s" % url)
                parts.append(chunk)
            return b"".join(parts), getattr(resp, "headers", {}) or {}
        finally:
            close = getattr(resp, "close", None)
            if close:
                close()

    def get(self, url):
        """Return ``(body_bytes, headers)``, or ``None`` on HTTP 404.

        Retries transient failures; refusals and oversize bodies are final.
        """
        last = None
        for attempt in range(self.retries + 1):
            try:
                return self._once(url)
            except TransientError as exc:
                last = str(exc)
                if attempt < self.retries:
                    self.sleep(self.backoff_s * (attempt + 1))
        raise DataError("failed after retries: %s" % last)
