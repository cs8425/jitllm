// The side panel: the engine's live numbers, the devices, the models and the
// sampling settings. Nothing here talks to the page's chat; it reads `live`.
import { h, svg, clear } from "./dom.ts";
import { live, subscribe, refreshLists, refreshFiles, refreshPlacement, changed, Series } from "./live.ts";
import { models, placement, sessions, errText } from "./api.ts";
import { BlockLocation, BlockKind, DeviceKind, ExecutionMode } from "./gen/jitllm/v1/common_pb";
import type { Device } from "./gen/jitllm/v1/device_pb";
import { settings, saveSettings } from "./store.ts";
import { human } from "./chat.ts";

export interface Ctx {
  model(): string;
  toast(msg: string, bad?: boolean): void;
  setModel(id: string): void;
}

const n = (x: bigint | number | undefined) => Number(x ?? 0);
const fmt = (x: number, d = 1) => (x >= 100 ? x.toFixed(0) : x.toFixed(d));

function kv(k: string, v: string | Node, tip = ""): HTMLElement {
  return h("div", { class: "kv", title: tip }, h("span", { class: "k", text: k }), h("span", { class: "v" }, v));
}
function card(title: string, ...kids: (Node | false | null)[]): HTMLElement {
  return h("section", { class: "card" }, h("h3", { text: title }), ...kids);
}
function meter(frac: number, cls = ""): HTMLElement {
  const f = Math.max(0, Math.min(1, isFinite(frac) ? frac : 0));
  return h("div", { class: "meter " + cls }, h("i", { style: { width: (f * 100).toFixed(1) + "%" } }));
}

export function spark(s: Series, w = 320, hgt = 64): SVGElement {
  const el = svg("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "spark", preserveAspectRatio: "none" });
  const v = s.v;
  if (v.length < 2) {
    el.append(svg("line", { x1: 0, y1: hgt - 1, x2: w, y2: hgt - 1, class: "base" }));
    return el;
  }
  const peak = Math.max(s.peak, 1e-9);
  const step = w / (Math.max(v.length, 40) - 1);
  const x0 = 0;
  const pts = v.map((y, i) => [x0 + i * step, hgt - 3 - (y / peak) * (hgt - 8)] as const);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  el.append(
    svg("path", { d: `${line} L${pts[pts.length - 1][0].toFixed(1)} ${hgt} L${x0.toFixed(1)} ${hgt} Z`, class: "area" }),
    svg("path", { d: line, class: "line" }),
    svg("circle", { cx: pts[pts.length - 1][0], cy: pts[pts.length - 1][1], r: 3, class: "dot" }),
  );
  return el;
}

let tab = settings.tab;
let root: HTMLElement;
let body: HTMLElement;
let ctx: Ctx;
let modelsSig = "";
const conv = new Map<string, string>(); // path -> conversion progress text

const TABS: [string, string][] = [["engine", "Engine"], ["devices", "Devices"], ["models", "Models"], ["settings", "Sampling"]];

export function mount(el: HTMLElement, c: Ctx) {
  root = el; ctx = c;
  const bar = h("nav", { class: "tabs", attrs: { role: "tablist" } });
  body = h("div", { class: "pbody" });
  const draw = () => {
    clear(bar);
    for (const [id, label] of TABS) bar.append(h("button", { class: "tab" + (id === tab ? " on" : ""), text: label, attrs: { role: "tab", "aria-selected": String(id === tab) }, on: { click: () => { tab = id; settings.tab = id; saveSettings(); modelsSig = ""; draw(); if (id === "models") refreshFiles(); render(true); } } }));
  };
  el.append(bar, body);
  draw();
  subscribe(() => render(false));
  render(true);
}
export function showTab(id: string) { if (id === "models") refreshFiles(); tab = id; settings.tab = id; saveSettings(); modelsSig = ""; root.querySelectorAll(".tab").forEach((b, i) => b.classList.toggle("on", TABS[i][0] === id)); render(true); }

function render(force: boolean) {
  if (tab === "engine") engine(force);
  else if (tab === "devices") deviceTab();
  else if (tab === "models") modelsTab(force);
  else if (force) settingsTab();
}

// ---- engine ---------------------------------------------------------------

let engineBuilt = false;
let dyn: HTMLElement, blocksBox: HTMLElement, ctl: HTMLElement;
let placeTimer = 0;

function engine(force: boolean) {
  if (force || !engineBuilt || !body.contains(dyn)) {
    clear(body);
    dyn = h("div", { class: "stack" });
    blocksBox = h("div", {});
    ctl = h("div", { class: "ctl" });
    body.append(dyn, card("Blocks", blocksBox, ctl));
    engineBuilt = true;
    refreshPlacement(ctx.model());
    clearInterval(placeTimer);
    placeTimer = window.setInterval(() => { if (tab === "engine") refreshPlacement(ctx.model()); }, 10000);
  }
  const id = ctx.model();
  const ms = live.stats?.models.find((m) => m.modelId === id);
  const info = live.list?.loaded.find((m) => m.modelId === id);
  const L = live.last;
  const rate = live.streaming ? live.decode.last : L.tps;
  const bw = rate && L.bytesPerToken ? (rate * L.bytesPerToken) / 1e9 : L.gbps;

  const hero = card("Throughput",
    h("div", { class: "hero" }, h("b", { text: rate ? fmt(rate) : "0.0" }), h("span", { text: " tok/s decode" }), live.streaming ? h("em", { class: "live", text: "live" }) : null),
    spark(live.decode),
    h("div", { class: "kvs" },
      kv("peak", live.decode.peak ? fmt(live.decode.peak) : "–"), kv("mean", live.decode.mean ? fmt(live.decode.mean) : "–"),
      kv("prompt", L.promptTps ? fmt(L.promptTps, 0) + " tok/s" : "–", "tokens read per second while prefilling"),
      kv("TTFT", L.ttft ? Math.round(L.ttft) + " ms" : "–", "time to first token, measured by this page"),
      kv("bandwidth", bw ? bw.toFixed(1) + " GB/s" : "–", "decode rate times bytes read per token"),
      kv("per token", L.bytesPerToken ? human(L.bytesPerToken) : "–")));

  const batch = ms?.batch;
  const sess = live.stats?.sessions.filter((s) => s.modelId === id) ?? [];
  const pos = Math.max(L.position, ...sess.map((s) => s.position), 0);
  const ctxLen = info?.contextLength ?? 0;
  const model = card("Model " + (id || ""),
    info ? h("div", { class: "kvs" },
      kv("architecture", info.architecture), kv("blocks", String(info.blockCount)),
      kv("width", String(info.embeddingDim)), kv("vocab", String(info.vocabSize)),
      info.expertCount ? kv("experts", `${info.expertsUsed} of ${info.expertCount}`) : null,
      info.linearBlocks ? kv("attn / linear", `${info.attentionBlocks} / ${info.linearBlocks}`) : null) : h("p", { class: "dim", text: id ? "Reading the model's details…" : "No model selected." }),
    ctxLen ? h("div", { class: "ctxrow" }, kv("context", `${pos} / ${ctxLen}`), meter(pos / ctxLen)) : null,
    ms ? h("div", { class: "kvs" },
      kv("generated", n(ms.tokensGenerated).toLocaleString()), kv("prefilled", n(ms.tokensPrefilled).toLocaleString()),
      kv("sessions", String(ms.sessions)),
      batch ? kv("batch rows", `${batch.liveRows} live · ${batch.waiting} waiting`, "rows stepping together, requests queued") : null,
      L.cached ? kv("prefix cache", `${L.cached} of ${L.prompt} tokens`, "prompt tokens answered from cached keys and values") : null) : null);

  const r = ms?.residency;
  const pager = r ? card("Pager",
    h("div", { class: "ctxrow" },
      kv("resident", `${r.resident?.human ?? "0"} / ${r.budget?.human ?? "0"}`), meter(n(r.resident?.bytes) / Math.max(1, n(r.budget?.bytes)), r.fits ? "" : "warn")),
    h("div", { class: "kvs" },
      kv("blocks", `${r.residentBlocks} of ${r.blocks}`), kv("frames", String(r.frames)),
      kv("page-ins", n(r.pageIns).toLocaleString()), kv("evictions", n(r.evictions).toLocaleString()),
      kv("read", `${r.bytesRead?.human ?? "0"} in ${n(r.readRequests)} reads`)),
    h("p", { class: r.fits ? "dim small" : "warn small", text: r.fits ? "The model fits in its page budget; nothing is evicted." : "Over budget: blocks are evicted and read back as needed." })) : null;

  dyn.replaceChildren(hero, model, ...(pager ? [pager] : []));
  drawBlocks();
}

function drawBlocks() {
  const p = live.place;
  clear(blocksBox);
  clear(ctl);
  if (!live.session || !p) {
    blocksBox.append(h("p", { class: "dim small", text: "A block map needs a session on the model: it holds the model's place on the host and the cards." }));
    if (ctx.model()) ctl.append(h("button", { class: "btn", text: "Open a session", on: { click: async () => {
      try { await sessions.createSession({ modelId: ctx.model() }); await refreshPlacement(ctx.model()); } catch (e) { ctx.toast(errText(e), true); }
    } } }));
    return;
  }
  const nd = p.deviceBlockCount, nb = p.blocks.length;
  blocksBox.append(
    h("div", { class: "seam" }, h("b", { text: String(nd) }), ` of ${nb} blocks on a device`,
      p.headOnDevice ? h("span", { class: "chip", text: "head on device" }) : null),
    h("div", { class: "blocks" }, ...p.blocks.map((b) => h("i", {
      class: "blk " + (b.location === BlockLocation.DEVICE ? "dev" : "host") + (b.kind === BlockKind.LINEAR ? " lin" : "") + (b.kind === BlockKind.VISION ? " vis" : "") + (b.hostResident ? "" : " cold"),
      title: `block ${b.index} · ${BlockKind[b.kind].toLowerCase()} · ${b.location === BlockLocation.DEVICE ? b.deviceId : "host"} · ${b.bytes?.human ?? ""}`,
    }))),
    h("div", { class: "legend" }, h("span", {}, h("i", { class: "blk host" }), " host"), h("span", {}, h("i", { class: "blk dev" }), " device"), h("span", {}, h("i", { class: "blk lin" }), " linear")),
    ...p.declines.slice(0, 3).map((d) => h("p", { class: "warn small", text: `block ${d.blockIndex}: ${d.reason}` })));
  const card_ = live.devs.find((d) => d.kind !== DeviceKind.HOST && d.available);
  const move = async (first: number, last: number, toHost: boolean) => {
    try {
      const r = await placement.relocateBlocks({ sessionId: live.session!.sessionId, firstBlock: first, lastBlock: last, toHost, toDeviceId: toHost ? "" : card_!.ref!.id });
      if (r.declined[0]) ctx.toast(r.declined[0].reason, true);
    } catch (e) { ctx.toast(errText(e), true); }
    await refreshPlacement(ctx.model());
  };
  const b = (label: string, on: boolean, f: () => void, tip: string) => h("button", { class: "btn sm", text: label, title: tip, disabled: !on, on: { click: f } });
  ctl.append(
    b("−1", nd > 0, () => move(nd - 1, nd - 1, true), "bring the last device block home"),
    b("+1", !!card_ && nd < nb, () => move(nd, nd, false), card_ ? `move the next block to ${card_.name}` : "no accelerator on this machine"),
    b("All host", nd > 0, () => move(0, nb - 1, true), "run every block on the host"),
    b("All device", !!card_ && nd < nb, () => move(0, nb - 1, false), "run every block on the accelerator"),
    h("button", { class: "btn sm ghost", text: "Close session", on: { click: async () => { try { await sessions.closeSession({ sessionId: live.session!.sessionId }); } catch (e) { ctx.toast(errText(e), true); } live.place = null; live.session = null; changed(); } } }));
}

// ---- devices --------------------------------------------------------------

function deviceCard(d: Device): HTMLElement {
  const total = n(d.totalMemory?.bytes), free = n(d.freeMemory?.bytes);
  const kind = d.kind === DeviceKind.HOST ? "host" : d.kind === DeviceKind.INTEGRATED ? "integrated" : "discrete";
  return h("section", { class: "card dev" + (d.available ? "" : " off") },
    h("h3", {}, d.name || d.ref?.id || "device", h("span", { class: "chip", text: kind })),
    h("div", { class: "id", text: d.ref?.id ?? "" }),
    h("div", { class: "ctxrow" }, kv("memory", `${d.freeMemory?.human ?? "?"} free of ${d.totalMemory?.human ?? "?"}`), meter(total ? 1 - free / total : 0)),
    h("div", { class: "kvs" },
      kv("sessions", `${d.attachedSessions} attached · ${d.runningSessions} running`), kv("queued", String(d.queuedRequests)),
      kv("execution", d.execution === ExecutionMode.PARALLEL ? "parallel" : d.execution === ExecutionMode.SERIALISED ? "serialised" : "–", d.executionNote)),
    d.sameDeviceAs ? h("p", { class: "dim small", text: "Same physical device as " + d.sameDeviceAs }) : null,
    d.available ? null : h("p", { class: "warn small", text: d.unavailableReason || "unavailable" }),
    d.countsTowardHostBudget ? h("p", { class: "dim small", text: "Shares the host's memory; not a second pool." }) : null);
}

function deviceTab() {
  const host = live.stats?.host;
  const i = live.info;
  clear(body);
  body.append(h("div", { class: "stack" },
    host ? card("Host",
      h("div", { class: "ctxrow" }, kv("available", `${host.available?.human ?? "?"} of ${host.total?.human ?? "?"}`), meter(1 - n(host.available?.bytes) / Math.max(1, n(host.total?.bytes)))),
      h("div", { class: "kvs" },
        kv("weight budget", host.weightBudget?.human ?? "?"), kv("decode cores", String(host.decodeCores)),
        kv("performance", String(host.performanceCores)), kv("efficiency", String(host.efficiencyCores)))) : null,
    ...live.devs.map(deviceCard),
    live.devs.length ? null : h("p", { class: "dim", text: "No devices reported." }),
    i ? card("Server", h("div", { class: "kvs" },
      kv("version", i.version + (i.modified ? "+" : "")), kv("platform", `${i.os}/${i.arch}`),
      kv("backends", i.availableBackends.join(", ") || "host"), kv("uptime", uptime(n(i.uptimeMillis))),
      kv("models", String(i.loadedModels)), kv("sessions", String(i.openSessions))),
      h("div", { class: "id", text: i.modelDirectory })) : null));
}
function uptime(ms: number) {
  const s = Math.floor(ms / 1000);
  return s < 90 ? s + " s" : s < 5400 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h";
}

// ---- models ---------------------------------------------------------------

let filter = "";
let loadOpts = { device: "", budgetGiB: "", maxBlocks: "" };

function modelsTab(force: boolean) {
  const m = live.files;
  const sig = JSON.stringify([live.scanning, live.list?.loaded.map((x) => x.modelId), m?.files.map((f) => [f.path, f.modelId, f.isContainer]), live.devs.map((d) => d.ref?.id), [...conv.entries()], ctx.model()]);
  if (!force && sig === modelsSig) return;
  if (!force && body.contains(document.activeElement) && document.activeElement instanceof HTMLInputElement) return; // do not rebuild under typing
  modelsSig = sig;
  clear(body);
  if (!m) { body.append(h("p", { class: "dim pad", text: live.scanning ? "Reading the model folder (this can take a moment on a busy machine)…" : "The model folder has not been read yet." }), h("button", { class: "btn", text: "Read it", on: { click: () => refreshFiles() } })); return; }
  const act = async (b: HTMLButtonElement, label: string, run: () => Promise<unknown>) => {
    b.disabled = true; b.textContent = label;
    try { await run(); } catch (e) { ctx.toast(errText(e), true); }
    await refreshFiles();
    modelsSig = "";
    render(true);
  };
  const loaded = (live.list?.loaded ?? m.loaded).map((x) => {
    const on = x.modelId === ctx.model();
    const un = h("button", { class: "btn sm ghost", text: "Unload" });
    un.addEventListener("click", () => act(un, "Unloading", () => models.unloadModel({ modelId: x.modelId })));
    return h("div", { class: "row" + (on ? " on" : "") , on: { click: (e: Event) => { if (e.target === un) return; ctx.setModel(x.modelId); } } },
      h("div", { class: "grow" }, h("b", { text: x.modelId }), h("div", { class: "dim small", text: `${x.architecture} · ${x.blockCount} blocks · ctx ${x.contextLength}` })),
      on ? h("span", { class: "chip ok", text: "active" }) : null, un);
  });
  const files = m.files.filter((f) => f.modelId === "" && !f.name.startsWith("mmproj-") && f.name.toLowerCase().includes(filter.toLowerCase()));
  const rows = files.map((f) => {
    const b = h("button", { class: "btn sm", text: f.isContainer ? "Load" : "Convert" });
    const prog = conv.get(f.path);
    if (f.isContainer) b.addEventListener("click", () => act(b, "Loading", () => models.loadModel({
      path: f.path,
      deviceIds: loadOpts.device ? [loadOpts.device] : [],
      pageBudgetBytes: loadOpts.budgetGiB ? BigInt(Math.round(Number(loadOpts.budgetGiB) * 2 ** 30)) : 0n,
      ...(loadOpts.maxBlocks !== "" ? { maxDeviceBlocks: Number(loadOpts.maxBlocks) } : {}),
    }).then(() => ctx.toast("Loaded " + f.name))));
    else b.addEventListener("click", async () => {
      const out = f.path.replace(/\.[^./]+$/, "") + ".jlm";
      conv.set(f.path, "starting"); modelsSig = ""; render(true);
      try {
        for await (const r of models.convert({ sourcePath: f.path, outputPath: out })) { conv.set(f.path, `${r.phase} ${Math.round(r.fraction * 100)}%`); modelsSig = ""; render(true); }
        ctx.toast("Converted " + f.name);
      } catch (e) { ctx.toast(errText(e), true); }
      conv.delete(f.path); await refreshFiles(); modelsSig = ""; render(true);
    });
    return h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "fname", text: f.name }), h("div", { class: "dim small", text: [f.size?.human, f.isContainer ? "container" : "source: converts first"].filter(Boolean).join(" · ") })),
      prog ? h("span", { class: "chip", text: prog }) : b);
  });
  const sel = h("select", { on: { change: (e: any) => (loadOpts.device = e.target.value) } }, h("option", { value: "", text: "Auto" }),
    ...live.devs.map((d) => h("option", { value: d.ref?.id ?? "", text: d.name || d.ref?.id || "" })));
  sel.value = loadOpts.device;
  body.append(h("div", { class: "stack" },
    card("Loaded", ...(loaded.length ? loaded : [h("p", { class: "dim small", text: "Nothing loaded. Pick a model file below." })])),
    card("Load options",
      h("label", {}, "Run on", sel),
      h("div", { class: "two" },
        h("label", {}, "Page budget (GiB)", h("input", { value: loadOpts.budgetGiB, placeholder: "auto", inputMode: "decimal", on: { input: (e: any) => (loadOpts.budgetGiB = e.target.value) } })),
        h("label", {}, "Device blocks", h("input", { value: loadOpts.maxBlocks, placeholder: "auto", inputMode: "numeric", on: { input: (e: any) => (loadOpts.maxBlocks = e.target.value) } }))),
      h("p", { class: "dim small", text: "Applied to the next load. Leave blank to let the engine place and budget itself." })),
    card("Model files", h("div", { class: "filehead" }, h("input", { class: "search", placeholder: "Filter", value: filter, on: { input: (e: any) => { filter = e.target.value; modelsSig = ""; render(true); (body.querySelector(".search") as HTMLInputElement)?.focus(); } } }), h("button", { class: "btn sm", text: live.scanning ? "Reading…" : "Refresh", title: "Read the model folder again", disabled: live.scanning, on: { click: () => refreshFiles() } })),
      ...(rows.length ? rows : [h("p", { class: "dim small", text: "No files here. " + (m.directory ? "Folder: " + m.directory : "") })]))));
}

// ---- sampling -------------------------------------------------------------

type Key = keyof typeof settings;
function field(label: string, k: Key, ph: string, tip = ""): HTMLElement {
  return h("label", { title: tip }, label, h("input", { value: String(settings[k]), placeholder: ph, inputMode: "decimal", on: { input: (e: any) => { (settings as any)[k] = e.target.value; saveSettings(); } } }));
}
function toggle(label: string, k: Key, tip = ""): HTMLElement {
  return h("label", { class: "tog", title: tip }, h("input", { type: "checkbox", checked: settings[k] as boolean, on: { change: (e: any) => { (settings as any)[k] = e.target.checked; saveSettings(); document.dispatchEvent(new CustomEvent("jitllm-settings")); } } }), label);
}
function settingsTab() {
  clear(body);
  body.append(h("div", { class: "stack" },
    card("Chat",
      toggle("Chat template", "chat", "Off sends the prompt as a raw completion, the way the benchmarks run."),
      toggle("Show thinking", "showThink", "Open <think> blocks while a reply streams."),
      toggle("Larger text", "bigText"),
      h("label", {}, "System prompt", h("textarea", { rows: 4, value: settings.system, placeholder: "Optional", on: { input: (e: any) => { settings.system = e.target.value; saveSettings(); } } }))),
    card("Sampling", h("p", { class: "dim small", text: "Blank keeps the engine's default. Temperature 0 is greedy." }),
      h("div", { class: "two" }, field("Temperature", "temperature", "default"), field("Top p", "topP", "off"), field("Top k", "topK", "off"), field("Min p", "minP", "off"),
        field("Repeat penalty", "repeatPenalty", "off"), field("Presence", "presence", "off"), field("Frequency", "frequency", "off"), field("Seed", "seed", "random"))),
    card("Output", field("Max tokens", "maxTokens", "1024"),
      h("label", {}, "Stop strings, one per line", h("textarea", { rows: 2, value: settings.stop, on: { input: (e: any) => { settings.stop = e.target.value; saveSettings(); } } })))));
}
