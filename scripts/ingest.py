import json
from datasets import load_dataset

# (exact title in dataset, status, replaced_by)
ACTS = [
    ("Code on Wages, 2019", "active", ""),
    ("Code on Social Security Act, 2020", "active", ""),
    ("Indian Contract Act, 1872", "active", ""),
    ("Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013", "active", ""),
    ("Payment of Gratuity Act, 1972", "repealed", "Code on Social Security, 2020"),
    ("Maternity Benefit Act, 1961", "repealed", "Code on Social Security, 2020"),
    ("Employees Provident Funds and Miscellaneous Provisions Act, 1952", "repealed", "Code on Social Security, 2020"),
    ("Payment of Wages Act, 1936", "repealed", "Code on Wages, 2019"),
    ("Minimum Wages Act, 1948", "repealed", "Code on Wages, 2019"),
    ("Industrial Disputes Act, 1947", "repealed", "Industrial Relations Code, 2020"),
    ("Industrial Employment (Standing Orders) Act, 1946", "repealed", "Industrial Relations Code, 2020"),
]
meta = {t: (s, r) for t, s, r in ACTS}

df = load_dataset("mratanusarkar/Indian-Laws", split="train").to_pandas()
missing = set(meta) - set(df["act_title"])
assert not missing, f"Titles not found: {missing}"
df = df[df["act_title"].isin(meta)]

def split_long(text, size=1500, overlap=200):
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out

rows = []
for r in df.to_dict("records"):
    status, replaced_by = meta[r["act_title"]]
    for piece in split_long(r["law"]):
        rows.append({
            "id": len(rows),
            "act": r["act_title"],
            "section": r["section"],
            "status": status,
            "replaced_by": replaced_by,
            "text": piece,
        })

with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(len(rows), "chunks saved")
for t in meta:
    print(f"  {sum(1 for r in rows if r['act'] == t):4d}  {t}")