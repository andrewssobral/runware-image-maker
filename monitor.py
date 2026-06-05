#! /usr/bin/env python3
import time

import httpx

while True:
    time.sleep(0.5)
    print(f"{httpx.get('http://localhost:12345/memory').json() / 1024**3:.2f} GiB")  # noqa: T201
