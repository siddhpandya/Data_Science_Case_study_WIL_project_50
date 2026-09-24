import requests

target_url = "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print(f"Querying Wayback availability for:\n{target_url}\n")

# Check CDX API for snapshots
cdx_url = f"https://web.archive.org/cdx/search/cdx?url={target_url}&output=json&fl=timestamp,original,statuscode,mimetype&filter=statuscode:200"

try:
    r = requests.get(cdx_url, headers=headers, timeout=25)
    data = r.json()
    print(f"Total snapshots found: {len(data) - 1}")
    if len(data) > 1:
        print("\nHeader:", data[0])
        # Sample across range
        rows = data[1:]
        print(f"First snapshot: {rows[0]}")
        print(f"Middle snapshot: {rows[len(rows)//2]}")
        print(f"Last snapshot: {rows[-1]}")
        
        # Look for a 2022/early 2023 snapshot (during relaxation)
        early_2023 = [row for row in rows if row[0].startswith("202301") or row[0].startswith("202302") or row[0].startswith("2022")]
        print(f"Snapshots in 2022/early 2023: {len(early_2023)}")
        if early_2023:
            print("Early candidate:", early_2023[-1])
            
        # Look for late 2023 / 2024 snapshot (post-cap restoration)
        late_2023 = [row for row in rows if row[0].startswith("202308") or row[0].startswith("202309") or row[0].startswith("202310") or row[0].startswith("2024")]
        print(f"Snapshots in late 2023/2024: {len(late_2023)}")
        if late_2023:
            print("Late candidate:", late_2023[0])
            
except Exception as e:
    print(f"Error querying CDX: {e}")
