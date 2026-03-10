import requests
from bs4 import BeautifulSoup
import csv

URL = "https://eecsis.mit.edu/whos_taken_what.html"

def fetch_page(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def parse_table(html, major):
    soup = BeautifulSoup(html, "html.parser")

    anchor = soup.find("a", attrs={"name": f"{major}"})
    if anchor is None:
        raise RuntimeError(f"Could not find {major} section anchor")

    table = anchor.find_next("table")
    if table is None:
        raise RuntimeError("Could not find table after section anchor")

    thead = table.find("thead")
    header_cells = thead.find_all("tr")[-1].find_all(["th", "td"])
    headers = [h.get_text(strip=True) for h in header_cells]
    headers.insert(0, "Subject")

    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        subject_cell = tds[0]
        link = subject_cell.find("a")
        subject_text = link.get_text(" ", strip=True) if link else subject_cell.get_text(" ", strip=True)
        subject_number = subject_text.split(" ")[0]

        counts = [td.get_text(strip=True) for td in tds[1:]]
        # counts = [int(num) for num in counts]

        row = [subject_number] + counts
        rows.append(row)

    max_len = max(len(headers), *(len(r) for r in rows))
    headers = headers[:max_len] + [""] * (max_len - len(headers))
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    return headers, rows

def save_to_csv(headers, rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def main():
    html = fetch_page(URL)
    major = "6-14"
    headers, rows = parse_table(html, major)
    save_to_csv(headers, rows, f"data/{major}_subjects.csv")
    print(f"Wrote {len(rows)} rows to {major}_subjects.csv")

if __name__ == "__main__":
    main()
