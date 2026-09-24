import os
import shutil
import threading
import zipfile
import pandas as pd
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

CSV_FILE = "ml_features.csv"
WORKERS = 256
TIMEOUT = (3, 5)

LOCAL_DIR = "/content/html_downloads"
HTML_DIR = os.path.join(LOCAL_DIR, "all_html")
DRIVE_DIR = "html_zips"

ZIP_SIZE = 400 * 1024 * 1024

df = pd.read_csv(CSV_FILE)

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)

local = threading.local()

def get_session():
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.headers.update({"User-Agent": "Mozilla/5.0"})
    return local.session

def connection_error_type(e):
    s = repr(e).lower()

    if "nameresolutionerror" in s or "name or service not known" in s or "nodename nor servname" in s or "no address associated with hostname" in s:
        return "dns_error"

    if "connection refused" in s:
        return "connection_refused"

    if "connection reset" in s:
        return "connection_reset"

    if "remotedisconnected" in s:
        return "remote_disconnected"

    if "network is unreachable" in s:
        return "network_unreachable"

    if "newconnectionerror" in s:
        return "new_connection_error"

    return "connection_error_other"

def process(i, url):
    try:
        r = get_session().get(
            url,
            timeout=TIMEOUT
        )

        if r.status_code != 200:
            return i, str(r.status_code)

        with open(os.path.join(HTML_DIR, f"{i:06d}.txt"), "wb") as f:
            f.write(r.content)

        return i, "200"

    except requests.exceptions.ConnectTimeout:
        return i, "connect_timeout"
    except requests.exceptions.ReadTimeout:
        return i, "read_timeout"
    except requests.exceptions.SSLError:
        return i, "ssl_error"
    except requests.exceptions.ConnectionError as e:
        return i, connection_error_type(e)
    except Exception as e:
        return i, f"other_error:{type(e).__name__}"

status_counts = Counter()
failed_by_source = Counter()
statuses = [None] * len(df)

rows = list(enumerate(df["url"]))
rows.sort(key=lambda x: urlparse(str(x[1])).netloc)

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [
        executor.submit(process, i, url)
        for i, url in rows
    ]

    with tqdm(total=len(df)) as pbar:
        for future in as_completed(futures):
            i, status = future.result()

            statuses[i] = status
            status_counts[status] += 1

            if status != "200":
                failed_by_source[df.at[i, "source"]] += 1

            ok = status_counts["200"]
            failed = sum(status_counts.values()) - ok

            pbar.update(1)
            pbar.set_postfix(
                ok=ok,
                failed=failed,
                majestic_fail=failed_by_source["majestic_million"],
                verified_fail=failed_by_source["verified_online"]
            )

df["fetch_status"] = statuses
df.to_csv("fetch_status.csv", index=False)

files = sorted(os.listdir(HTML_DIR))

part = 1
zip_path = os.path.join(LOCAL_DIR, f"html_part_{part:02d}.zip")
z = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

for filename in tqdm(files, desc="Zipping"):
    path = os.path.join(HTML_DIR, filename)
    z.write(path, filename)

    if os.path.getsize(zip_path) >= ZIP_SIZE:
        z.close()

        shutil.copy2(
            zip_path,
            os.path.join(DRIVE_DIR, os.path.basename(zip_path))
        )

        os.remove(zip_path)

        part += 1
        zip_path = os.path.join(
            LOCAL_DIR,
            f"html_part_{part:02d}.zip"
        )

        z = zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        )

z.close()

if os.path.getsize(zip_path) > 0:
    shutil.copy2(
        zip_path,
        os.path.join(DRIVE_DIR, os.path.basename(zip_path))
    )
    os.remove(zip_path)

shutil.rmtree(HTML_DIR)

print("\nFetch status:")
for status, count in status_counts.most_common():
    print(f"{status}: {count}")

print("\nFailed by source:")
print("majestic_million:", failed_by_source["majestic_million"])
print("verified_online:", failed_by_source["verified_online"])
