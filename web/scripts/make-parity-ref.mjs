// Usage: node scripts/make-parity-ref.mjs
// Cuts the verse line out of the test reference (PyTorch logits on fp16-rounded weights)
// into a small file the page can fetch to re-run the parity check live.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(resolve(here, "../tests/fixtures/reference.json"), "utf8"));
const v = ref.float16.verse;
const out = { text: v.text, ids: v.ids, logits: v.logits };
const path = resolve(here, "../public/model/parity-verse.json");
writeFileSync(path, JSON.stringify(out));
console.log("written", path, v.ids.length, "positions");
