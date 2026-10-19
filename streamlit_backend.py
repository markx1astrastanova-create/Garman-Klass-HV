from urllib.parse import parse_qs, urlsplit

from fastapi import HTTPException


def dispatch(request, backend):
    """Expose only the two read-only operations used by the React app."""
    request_id = request.get("id")
    if not isinstance(request_id, str) or len(request_id) > 100:
        raise ValueError("Invalid request ID")
    try:
        path = request.get("path")
        if not isinstance(path, str) or len(path) > 500:
            raise HTTPException(status_code=400, detail="Invalid request path")
        url = urlsplit(path)
        if url.scheme or url.netloc or url.fragment:
            raise HTTPException(status_code=400, detail="Invalid request path")
        if url.path == "/api/tickers":
            body = backend.get_tickers()
        elif url.path == "/api/volatility":
            query = parse_qs(url.query)
            ticker = query.get("ticker", ["JKSE"])[0]
            try:
                window = int(query.get("window", ["60"])[0])
            except ValueError:
                raise HTTPException(status_code=422, detail="Window must be an integer")
            if not 10 <= window <= 200 or len(ticker) > 50:
                raise HTTPException(status_code=422, detail="Invalid ticker or window (10–200)")
            body = backend.get_volatility(ticker=ticker, window=window)
        else:
            raise HTTPException(status_code=404, detail="Unknown operation")
        return {"id": request_id, "status": 200, "body": body}
    except HTTPException as exc:
        return {"id": request_id, "status": exc.status_code, "body": {"detail": exc.detail}}
