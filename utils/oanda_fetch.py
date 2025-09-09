import time, random, logging, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_oanda_session(token: str,
                       host: str = "https://api-fxpractice.oanda.com",
                       timeout=(3.05, 10)):
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.host = host.rstrip("/")
    s.timeout = timeout
    return s

def fetch_candles(session, instrument: str, granularity: str,
                  count: int = 800, price: str = "M"):
    """
    OANDA /v3/instruments/{instrument}/candles with retry + trimmed errors.
    Returns parsed JSON dict or raises RuntimeError with a concise message.
    """
    url = f"{session.host}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "count": count, "price": price}
    try:
        r = session.get(url, params=params, timeout=session.timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"OANDA network error: {e.__class__.__name__}: {e}") from e

    if 200 <= r.status_code < 300:
        return r.json()

    # if we get here, retries exhausted or non-retryable code
    body = (r.text or "")[:400].replace("\n", " ")
    raise RuntimeError(f"OANDA HTTP {r.status_code} {instrument}/{granularity}: {body}")
