import requests

target_url = "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print("Checking archive.org/wayback/available for temporary-relaxation...")

for ts in ["20220601", "20230101", "20230501", "20230901", "20240101"]:
    api = f"https://archive.org/wayback/available?url={target_url}&timestamp={ts}"
    try:
        r = requests.get(api, headers=headers, timeout=20)
        data = r.json()
        closest = data.get("archived_snapshots", {}).get("closest", {})
        print(f"{ts} -> available: {closest.get('available')}, ts: {closest.get('timestamp')}")
        if closest.get('url'):
            print(f"       url: {closest.get('url')}")
    except Exception as e:
        print(f"{ts} -> Error: {e}")
