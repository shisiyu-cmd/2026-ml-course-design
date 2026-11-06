"""Parallel segmented downloader (network is throttled per-connection)."""
import os, sys, time, threading, urllib.request

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
NTHREADS = 16
CHUNK = 4 * 1024 * 1024


def head_size(url):
    req = urllib.request.Request(url, headers=HDR, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"]), r.headers.get("Accept-Ranges", "")


def download(url, out, nthreads=NTHREADS, chunk=CHUNK):
    total, ranges = head_size(url)
    print(f"size: {total/1e6:.1f} MB   accept-ranges={ranges}", flush=True)
    nparts = (total + chunk - 1) // chunk
    parts = [bytearray()] * nparts
    lock = threading.Lock()
    done = [0]
    t_start = time.time()

    def worker(i):
        lo = i * chunk
        hi = min(lo + chunk, total) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers=dict(HDR, Range=f"bytes={lo}-{hi}"))
                with urllib.request.urlopen(req, timeout=60) as r:
                    buf = bytearray()
                    while True:
                        b = r.read(131072)
                        if not b:
                            break
                        buf += b
                if len(buf) == hi - lo + 1:
                    parts[i] = buf
                    with lock:
                        done[0] += len(buf)
                    return
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        print(f"  part {i} FAILED", flush=True)

    idx = list(range(nparts))
    ths = []
    running = []
    # simple pool
    queue = list(idx)
    qlock = threading.Lock()

    def run():
        while True:
            with qlock:
                if not queue:
                    return
                i = queue.pop(0)
            worker(i)

    for _ in range(nthreads):
        t = threading.Thread(target=run, daemon=True)
        t.start()
        ths.append(t)

    last = 0
    while any(t.is_alive() for t in ths):
        time.sleep(2)
        with lock:
            d = done[0]
        el = time.time() - t_start
        spd = (d - last) / 2 / 1e6
        last = d
        print(f"  {d/1e6:8.1f}/{total/1e6:.1f} MB  {100*d/total:5.1f}%  {spd:5.2f} MB/s", flush=True)
    for t in ths:
        t.join()

    with open(out, "wb") as fh:
        for p in parts:
            fh.write(p)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB) in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    download(url, out)
