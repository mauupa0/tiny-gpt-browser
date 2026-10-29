import { useCallback, useEffect, useRef, useState } from "react";
import { useReducedMotion } from "./state/motion";
import { useModel } from "./state/store";
import faultsData from "./generated/faults.json";
import headsData from "./generated/heads.json";
import type { HeadsData } from "./lib/heads";
import { Chapter } from "./ui/Chapter";
import { Hero, PRESETS, paceFor, type HeroSettings } from "./ui/Hero";
import { Atlas } from "./ui/Atlas";
import { BreakIt } from "./ui/BreakIt";
import { Proof } from "./ui/Proof";
import { Temperature } from "./ui/Temperature";
import type { LaneSpec } from "./worker/protocol";
import { Plate } from "./ui/Plate";

const BASE = `${import.meta.env.BASE_URL}model/`;
const STANZA = 320;
const HEADS = headsData as HeadsData;
const FAULTS = faultsData;

const CHAPTERS = [
  { id: "writing", label: "Writing" },
  { id: "inside", label: "Inside" },
  { id: "heads", label: "Heads" },
  { id: "temperature", label: "Temperature" },
  { id: "break", label: "Break it" },
  { id: "proof", label: "Proof" },
];

function useActiveChapter(): string {
  const [active, setActive] = useState("writing");
  useEffect(() => {
    const els = CHAPTERS.map((c) => document.getElementById(c.id)).filter((e): e is HTMLElement => !!e);
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: "-45% 0px -50% 0px" },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
  return active;
}

// The chapter links scroll sideways on a phone. Tell the CSS which edges still hide links, and keep
// the current chapter centred, moving only the strip (scrollIntoView would move the page too).
function useNavStrip(active: string, reduced: boolean) {
  const nav = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = nav.current;
    if (!el) return;
    const edges = () => {
      el.dataset.start = String(el.scrollLeft <= 1);
      el.dataset.end = String(el.scrollLeft + el.clientWidth >= el.scrollWidth - 1);
    };
    edges();
    el.addEventListener("scroll", edges, { passive: true });
    window.addEventListener("resize", edges);
    return () => {
      el.removeEventListener("scroll", edges);
      window.removeEventListener("resize", edges);
    };
  }, []);
  useEffect(() => {
    const el = nav.current;
    const link = el?.querySelector<HTMLElement>('a[aria-current="true"]');
    if (!el || !link || el.scrollWidth <= el.clientWidth + 1) return;
    const at = link.getBoundingClientRect().left - el.getBoundingClientRect().left + el.scrollLeft;
    el.scrollTo({ left: at - (el.clientWidth - link.offsetWidth) / 2, behavior: reduced ? "auto" : "smooth" });
  }, [active, reduced]);
  return nav;
}

export function App() {
  const model = useModel(BASE);
  const { state } = model;
  const reduced = useReducedMotion();
  const active = useActiveChapter();
  const nav = useNavStrip(active, reduced);
  const [prompt, setPrompt] = useState(PRESETS[0]);
  const [settings, setSettings] = useState<HeroSettings>({ temperature: 0.8, topK: 20, fast: false });

  const { setText, writeHero, finishHero, stop, setParams } = model;
  const ready = state.phase === "ready";
  const running = state.runs.hero.running;

  const lastPrompt = state.text.slice(0, state.promptLen).map((i) => state.meta?.chars[i] ?? "").join("");
  useEffect(() => {
    if (!ready || running || prompt === lastPrompt) return;
    const t = window.setTimeout(() => setText(prompt), 300);
    return () => window.clearTimeout(t);
  }, [ready, running, prompt, lastPrompt, setText]);

  useEffect(() => {
    if (running) setParams("hero", [settings.temperature], settings.topK, paceFor(settings.fast));
  }, [running, settings, setParams]);

  const onWrite = useCallback(
    (p: string | null) =>
      writeHero({ prompt: p, temperature: settings.temperature, topK: settings.topK, paceMs: paceFor(settings.fast), maxNew: p === null ? 220 : STANZA, endAtBreak: p === null ? 110 : 0 }),
    [writeHero, settings],
  );
  const onStop = useCallback(() => stop("hero"), [stop]);
  const { writeLanes } = model;
  const pace = paceFor(settings.fast);
  const onTemp = useCallback((lanes: LaneSpec[]) => writeLanes("temp", lanes, { topK: settings.topK, paceMs: pace, maxNew: 150 }), [writeLanes, settings.topK, pace]);
  const onBreak = useCallback((lanes: LaneSpec[]) => writeLanes("brk", lanes, { topK: settings.topK, paceMs: pace, maxNew: 150 }), [writeLanes, settings.topK, pace]);
  const stopTemp = useCallback(() => stop("temp"), [stop]);
  const stopBreak = useCallback(() => stop("brk"), [stop]);

  return (
    <>
      <div className="topbar">
        <div className="wrap topbar-in">
          <a className="mark" href="#writing">
            Tiny GPT
          </a>
          <nav className="nav" aria-label="Chapters" ref={nav}>
            {CHAPTERS.map((c) => (
              <a key={c.id} href={`#${c.id}`} aria-current={active === c.id ? "true" : undefined}>
                {c.label}
              </a>
            ))}
          </nav>
        </div>
      </div>
      {state.phase === "ready" && state.error && (
        <p className="banner" role="alert">
          The model worker reported a problem: {state.error}
        </p>
      )}
      <main>
        <Hero
          s={state}
          reduced={reduced}
          settings={settings}
          onSettings={setSettings}
          prompt={prompt}
          onPrompt={setPrompt}
          onWrite={onWrite}
          onStop={onStop}
          onFinish={finishHero}
          onSelect={model.select}
          onFollow={model.follow}
        />
        <Chapter
          id="inside"
          title="One letter, start to finish"
          lead={
            <>
              Every number below was computed by the model in this tab for the letter you picked in the poem. It shows what the model read, which of its 16 attention heads listened to which letters, and what each part
              pushed towards the answer.
            </>
          }
        >
          <Plate
            insp={state.inspection}
            busy={state.inspecting}
            phase={state.phase}
            chars={state.meta?.chars ?? []}
            heads={HEADS}
            idx={state.inspectIdx}
            textLen={state.text.length}
            nextId={state.text[state.inspectIdx + 1] ?? null}
            explain={state.explain}
            onExplain={model.setExplain}
            onSelect={model.select}
          />
        </Chapter>
        <Chapter
          id="heads"
          title="Sixteen habits"
          lead={
            <>
              Training gave each attention head its own way of reading. I measured them on {HEADS.windows} windows of the poem the model never saw: how far back each one looks, and how much worse the model gets when that head is switched
              off. The maps show what each head did for the letter you inspected.
            </>
          }
        >
          <Atlas insp={state.inspection} chars={state.meta?.chars ?? []} heads={HEADS} />
        </Chapter>
        <Chapter
          id="temperature"
          title="Temperature"
          lead={
            <>
              Temperature rescales the model&apos;s probabilities before it draws a letter. Low values make it take the favourite almost every time, high values let long shots through. Three writers read your opening line below and draw
              from the same random numbers, so the only difference between them is the temperature.
            </>
          }
        >
          <Temperature s={state} topK={settings.topK} paceMs={pace} reduced={reduced} onWrite={onTemp} onStop={stopTemp} />
        </Chapter>
        <Chapter
          id="break"
          title="Break it"
          lead={
            <>
              A test is only worth something if it fails when the code is wrong. Two classic transformer bugs are built into this model behind a switch. Dropping the 1/√d factor costs {(FAULTS.scale - FAULTS.none).toFixed(2)} nats per
              letter on held-out text; letting attention see one slot ahead costs {(FAULTS.mask - FAULTS.none).toFixed(3)}, which no reader would notice in the verse. The parity check against PyTorch catches both.
            </>
          }
        >
          <BreakIt s={state} temperature={settings.temperature} reduced={reduced} onWrite={onBreak} onStop={stopBreak} onAblation={model.setAblation} onParity={model.runParity} />
        </Chapter>
        <Chapter
          id="proof"
          title="The evidence"
          lead={<>The training run, where the parameters sit, the test suite and the speed on this device.</>}
        >
          <Proof ready={state.phase === "ready"} bench={state.bench} benchBusy={state.benchBusy} onBench={model.runBench} />
        </Chapter>
      </main>
      <footer className="foot">
        <div className="wrap foot-in">
          <p>
            Text: Adam Mickiewicz, <i>Pan Tadeusz</i>, from{" "}
            <a href="https://wolnelektury.pl/katalog/lektura/pan-tadeusz/" rel="noreferrer">
              Wolne Lektury
            </a>
            . The poem is in the public domain; the editorial notes were removed before training.
          </p>
          <p>Built by Oktawian Palikot. Training in PyTorch on CPU, inference in hand-written TypeScript, drawn on canvas.</p>
        </div>
      </footer>
    </>
  );
}
