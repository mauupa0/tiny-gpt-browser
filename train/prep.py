import re, json, collections, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
raw = (root / "data" / "raw.txt").read_text(encoding="utf-8")
body = raw.split("\n-----")[0]
start = body.index("Litwo! Ojczyzno")
head_cut = body[:start]
body = body[start:]
paras = re.split(r"\n\s*\n", body)
keep = []
for p in paras:
    lines = [l.strip() for l in p.split("\n") if l.strip()]
    if not lines:
        continue
    if len(lines) == 1 and (lines[0].startswith("Księga") or len(lines[0]) < 30 or (" \u2014 " in lines[0] and len(lines[0]) > 100)):
        continue
    keep.append("\n".join(lines))
text = "\n\n".join(keep)
text = text.replace("\u2014", "-").replace("\u2013", "-").replace("\r", "")
text = text.replace("…", "...").replace(" ", " ")
text = "Litwo! Ojczyzno" + text[len("Litwo! Ojczyzno"):]
cnt = collections.Counter(text)
rare = {c for c, n in cnt.items() if n < 20}
text = "".join(c for c in text if c not in rare)
(root / "data" / "corpus.txt").write_text(text, encoding="utf-8", newline="\n")
chars = sorted(set(text))
print(len(text), "chars", len(chars), "vocab", "dropped:", sorted(rare))
print(repr("".join(chars)))
print(text[:300])
