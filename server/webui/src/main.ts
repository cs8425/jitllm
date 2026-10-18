import { create } from "@bufbuild/protobuf";
import { GenerateRequestSchema } from "./gen/jitllm/v1/inference_pb";
import { FinishReason } from "./gen/jitllm/v1/common_pb";
import { h, clear } from "./dom.ts";
import { markdown } from "./md.ts";
import { parseStop, optNumber, requestMessages, splitThink, human } from "./chat.ts";
import { infer, errText, isUnauth, keepKey, hasKey, isCancel, models as modelsApi } from "./api.ts";
import { live, start, subscribe, refreshLists, changed, loadedIds } from "./live.ts";
import { settings, saveSettings, allChats, newChat, touch, dropChat, saveChats, type Chat, type Msg, type Meta } from "./store.ts";
import * as panel from "./panel.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- state ----------------------------------------------------------------

let chat: Chat = allChats()[0] ?? newChat("");
let busy: AbortController | null = null;
const pictures = new WeakMap<Msg, { data: Uint8Array; mediaType: string }[]>();
let pending: { data: Uint8Array; mediaType: string }[] = [];
const views = new Map<Msg, { update(): void; root: HTMLElement }>();
let stick = true;

const modelId = () => ($("model") as HTMLSelectElement | null)?.value || settings.model;

function toast(msg: string, bad = false) {
  const t = h("div", { class: "toast" + (bad ? " bad" : ""), text: msg });
  $("toasts").append(t);
  setTimeout(() => t.remove(), bad ? 6000 : 2600);
}

// ---- header ---------------------------------------------------------------

const header = $("top");
function drawHeader() {
  const sel = h("select", { id: "model", title: "Model", attrs: { "aria-label": "Model" }, on: { change: (e: any) => { settings.model = e.target.value; chat.model = e.target.value; saveSettings(); drawHeader(); changed(); } } });
  const loaded = loadedIds().map((modelId) => ({ modelId }));
  for (const m of loaded) sel.append(h("option", { value: m.modelId, text: m.modelId }));
  if (!loaded.length) sel.append(h("option", { value: "", text: live.list || live.stats ? "no model loaded" : "connecting…" }));
  const want = loaded.some((m) => m.modelId === settings.model) ? settings.model : chat.model && loaded.some((m) => m.modelId === chat.model) ? chat.model : loaded[0]?.modelId ?? "";
  sel.value = want;
  settings.model = want;
  const dotCls = live.connected ? "dot ok" : "dot bad";
  const rate = live.streaming ? live.decode.last : live.last.tps;
  header.replaceChildren(
    h("button", { class: "icon only-narrow", title: "Conversations (alt+L)", text: "☰", on: { click: () => drawer("chats") } }),
    h("div", { class: "brand", text: "jitllm" }),
    sel,
    h("div", { class: "seg", attrs: { role: "group", "aria-label": "Prompt mode" } },
      h("button", { class: settings.chat ? "on" : "", text: "Chat", title: "Apply the chat template", on: { click: () => setChat(true) } }),
      h("button", { class: settings.chat ? "" : "on", text: "Raw", title: "Complete the prompt as written", on: { click: () => setChat(false) } })),
    h("button", { class: "toggle" + (settings.showThink ? " on" : ""), text: "Thinking", title: "Show reasoning (alt+T)", on: { click: toggleThink } }),
    h("div", { class: "grow" }),
    h("div", { class: "chip rate" + (live.streaming ? " live" : ""), title: "Decode rate", text: rate ? `${rate.toFixed(1)} tok/s` : "idle" }),
    h("div", { class: "status", title: live.error || "connected" }, h("i", { class: dotCls }), h("span", { class: "dim", text: live.connected ? (live.info ? "v" + live.info.version : "connected") : live.error || "connecting" })),
    h("button", { class: "icon", title: "Commands (ctrl+K)", text: "⌘", on: { click: palette } }),
    h("button", { class: "icon" + (settings.panel ? " on" : ""), title: "Engine panel (alt+B)", text: "◧", on: { click: togglePanel } }));
}
function setChat(on: boolean) { settings.chat = on; saveSettings(); drawHeader(); toast(on ? "Chat template on" : "Raw completion"); }
function toggleThink() {
  settings.showThink = !settings.showThink; saveSettings(); drawHeader();
  document.querySelectorAll<HTMLDetailsElement>("details.think").forEach((d) => { d.open = settings.showThink; });
}
function togglePanel() { settings.panel = !settings.panel; saveSettings(); layout(); drawHeader(); if (window.innerWidth < 1100) drawerState(settings.panel ? "panel" : ""); }
function toggleChats() { settings.chats = !settings.chats; saveSettings(); layout(); if (window.innerWidth < 800) drawer("chats"); }
function layout() {
  const app = $("app");
  app.classList.toggle("no-panel", !settings.panel);
  app.classList.toggle("no-chats", !settings.chats);
  document.body.classList.toggle("big", settings.bigText);
}
let open = "";
function drawerState(which: string) {
  open = which;
  $("app").dataset.drawer = which;
  $("scrim").hidden = !which;
}
function drawer(which: string) { drawerState(open === which ? "" : which); }
$("scrim").addEventListener("click", () => drawerState(""));

// ---- chat list ------------------------------------------------------------

let chatFilter = "";
function drawChats() {
  const box = $("chats");
  const list = allChats().filter((c) => c.title.toLowerCase().includes(chatFilter.toLowerCase()));
  box.replaceChildren(
    h("div", { class: "side-head" }, h("button", { class: "btn wide", text: "+ New chat", title: "alt+N", on: { click: () => switchTo(newChatNow()) } })),
    h("input", { class: "search", placeholder: "Search chats", value: chatFilter, on: { input: (e: any) => { chatFilter = e.target.value; drawChats(); (box.querySelector(".search") as HTMLInputElement).focus(); } } }),
    h("div", { class: "clist" }, ...list.map((c) => h("div", { class: "citem" + (c === chat ? " on" : ""), title: c.title, on: { click: () => switchTo(c) } },
      h("div", { class: "ct", text: c.title }),
      h("div", { class: "cm", text: `${c.msgs.length} msg · ${c.model || "no model"} · ${ago(c.at)}` }),
      h("button", { class: "x", title: "Delete chat", text: "×", on: { click: (e: Event) => { e.stopPropagation(); removeChat(c); } } })))),
    ...(list.length ? [] : [h("p", { class: "dim small pad", text: "No chats yet." })]));
}
const ago = (t: number) => { const s = (Date.now() - t) / 1000; return s < 60 ? "now" : s < 3600 ? Math.floor(s / 60) + "m" : s < 86400 ? Math.floor(s / 3600) + "h" : Math.floor(s / 86400) + "d"; };
function newChatNow(): Chat {
  if (chat.msgs.length === 0) return chat;
  return newChat(modelId());
}
function switchTo(c: Chat) {
  if (busy) { toast("A reply is streaming: Esc stops it", true); return; }
  chat = c;
  if (c.model && loadedIds().includes(c.model)) settings.model = c.model;
  drawerState("");
  drawAll();
  $("prompt")?.focus();
}
function removeChat(c: Chat) {
  if (busy && c === chat) { toast("A reply is streaming: Esc stops it", true); return; }
  dropChat(c);
  if (c === chat) chat = allChats()[0] ?? newChat(modelId());
  drawAll();
}
function step(d: number) {
  const l = allChats();
  const i = l.indexOf(chat);
  if (!l.length) return;
  switchTo(l[(i + d + l.length) % l.length]);
}

// ---- transcript -----------------------------------------------------------

const log = $("log");
log.addEventListener("scroll", () => {
  stick = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  $("latest").hidden = stick;
});
$("latest").addEventListener("click", () => { stick = true; scrollEnd(); });
function scrollEnd() { if (stick) log.scrollTop = log.scrollHeight; $("latest").hidden = stick; }

function drawLog() {
  views.clear();
  log.replaceChildren();
  if (chat.msgs.length === 0) { log.append(emptyState()); return; }
  const wrap = h("div", { class: "thread" });
  chat.msgs.forEach((m, i) => wrap.append(msgView(m, i).root));
  log.append(wrap);
  stick = true;
  scrollEnd();
}

function emptyState(): HTMLElement {
  const ok = !!modelId();
  const sug = ["Explain how a KV cache speeds up decoding.", "Write a Go function that merges two sorted slices.", "Summarise the tradeoffs of paging model weights from disk.", "Give me a haiku about a warm laptop fan."];
  return h("div", { class: "empty" },
    h("div", { class: "mark", text: "jitllm" }),
    h("p", { class: "tag", text: "Every kernel generated for the machine in front of it." }),
    ok ? h("div", { class: "sug" }, ...sug.map((s) => h("button", { class: "sugg", text: s, on: { click: () => { const p = $<HTMLTextAreaElement>("prompt"); p.value = s; send(); } } })))
      : h("div", { class: "noload" }, h("p", { text: live.list || live.stats ? "No model is loaded." : "Connecting to the server…" }),
        live.list || live.stats ? h("button", { class: "btn", text: "Choose a model", on: { click: () => { settings.panel = true; layout(); drawHeader(); panel.showTab("models"); drawerState(window.innerWidth < 1100 ? "panel" : ""); } } }) : null),
    h("div", { class: "keys" }, ...[["Enter", "send"], ["Shift+Enter", "new line"], ["Esc", "stop"], ["Ctrl+K", "commands"], ["?", "all shortcuts"]].map(([k, v]) => h("span", {}, h("kbd", { text: k }), " " + v))));
}

function metaLine(m: Meta): string {
  if (m.error) return m.error;
  const bits: string[] = [];
  if (m.stopped) bits.push("stopped");
  else if (m.reason) bits.push(m.reason);
  if (m.tokens) bits.push(`${m.tokens} tokens`);
  if (m.tps) bits.push(`${m.tps.toFixed(1)} tok/s`);
  if (m.ttft) bits.push(`TTFT ${Math.round(m.ttft)} ms`);
  if (m.prompt) bits.push(`prompt ${m.prompt}${m.cached ? ` (${m.cached} cached)` : ""}`);
  if (m.gbps) bits.push(`${m.gbps.toFixed(1)} GB/s`);
  return bits.join(" · ");
}

function msgView(m: Msg, index: number): { update(): void; root: HTMLElement } {
  const root = h("article", { class: "m " + m.role });
  const v = { root, update() {} };
  views.set(m, v);
  if (m.role === "user") {
    const imgs = pictures.get(m) ?? [];
    root.append(h("div", { class: "bubble" },
      ...imgs.map((im) => h("img", { src: URL.createObjectURL(new Blob([im.data as BlobPart], { type: im.mediaType })), alt: "attached picture" })),
      !imgs.length && m.pics ? h("div", { class: "dim small", text: `${m.pics} picture(s)` }) : null,
      h("div", { class: "txt", text: m.content })),
      h("div", { class: "acts" },
        h("button", { text: "Copy", on: { click: () => copy(m.content) } }),
        h("button", { text: "Edit", title: "Take this back into the prompt", on: { click: () => editFrom(index) } })));
    return v;
  }
  const head = h("div", { class: "who" }, h("span", { class: "av", text: "j" }), h("b", { text: m.model || "assistant" }));
  const think = h("details", { class: "think" }) as HTMLDetailsElement;
  const tsum = h("summary", {});
  const tbody = h("div", { class: "tbody" });
  think.append(tsum, tbody);
  const ans = h("div", { class: "ans" });
  const meta = h("div", { class: "meta" });
  const acts = h("div", { class: "acts" });
  root.append(head, think, ans, meta, acts);
  let thinkOpen: boolean | null = null;
  let folded = false;
  v.update = () => {
    const p = splitThink(m.content);
    think.hidden = !p.think && !p.thinking;
    if (!think.hidden) {
      const first = thinkOpen === null;
      if (first) { thinkOpen = true; think.open = settings.showThink && !!busy && isLast(m); }
      // Once the answer starts, the reasoning folds away; the summary reopens it.
      if (!p.thinking && p.answer && busy && isLast(m) && !folded) { folded = true; think.open = false; }
      tsum.textContent = p.thinking ? "Thinking…" : `Thought for ${p.think.split(/\s+/).length} words`;
      tsum.className = p.thinking ? "pulse" : "";
      tbody.replaceChildren(markdown(p.think));
    }
    ans.replaceChildren(p.answer ? markdown(p.answer) : (busy && views.get(m) && isLast(m) && !p.thinking ? h("span", { class: "caret" }) : ""));
    if (isLast(m) && busy && p.answer) ans.lastElementChild?.append(h("span", { class: "caret" }));
    meta.textContent = m.meta ? metaLine(m.meta) : "";
    meta.className = "meta" + (m.meta?.error ? " bad" : "");
    acts.replaceChildren(
      h("button", { text: "Copy", on: { click: () => copy(p.answer || m.content) } }),
      ...(isLast(m) && !busy ? [h("button", { text: "Regenerate", on: { click: regenerate } })] : []));
  };
  v.update();
  return v;
}
const isLast = (m: Msg) => chat.msgs[chat.msgs.length - 1] === m;
async function copy(t: string) { try { await navigator.clipboard.writeText(t); toast("Copied"); } catch { toast("Clipboard unavailable", true); } }
function lastReply() { return [...chat.msgs].reverse().find((m) => m.role === "assistant"); }
function copyLast() { const m = lastReply(); if (m) copy(splitThink(m.content).answer || m.content); }

function editFrom(i: number) {
  if (busy) return;
  const m = chat.msgs[i];
  chat.msgs.length = i;
  touch(chat);
  $<HTMLTextAreaElement>("prompt").value = m.content;
  drawLog(); drawChats();
  $("prompt").focus();
}
function regenerate() {
  if (busy) return;
  const last = chat.msgs[chat.msgs.length - 1];
  if (last?.role === "assistant") chat.msgs.pop();
  const lastUser = chat.msgs[chat.msgs.length - 1];
  if (lastUser?.role !== "user") return;
  drawLog();
  run();
}

// ---- composer -------------------------------------------------------------

const comp = $("composer");
function drawComposer() {
  const prev = document.getElementById("prompt") as HTMLTextAreaElement | null;
  const draft = prev?.value ?? "";
  const ta = h("textarea", { id: "prompt", rows: 1, placeholder: "Message jitllm…   (/ for commands)", attrs: { "aria-label": "Message" }, value: draft, on: {
    input: () => { grow(); hints(); },
    keydown: (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
      else if (e.key === "ArrowUp" && ta.value === "") { const u = [...chat.msgs].reverse().find((m) => m.role === "user"); if (u) { e.preventDefault(); ta.value = u.content; grow(); } }
    },
    paste: (e: ClipboardEvent) => { const fs = [...(e.clipboardData?.files ?? [])].filter((f) => f.type.startsWith("image/")); if (fs.length) { e.preventDefault(); attach(fs); } },
  } });
  const file = h("input", { type: "file", id: "file", accept: "image/png,image/jpeg,image/webp,image/gif", multiple: true, hidden: true, on: { change: (e: any) => { attach([...e.target.files]); e.target.value = ""; } } });
  comp.replaceChildren(
    h("div", { id: "hints", hidden: true }),
    h("div", { id: "thumbs" }),
    h("div", { class: "box" },
      file,
      h("button", { class: "icon", title: "Attach pictures (or drop / paste them)", text: "＋", on: { click: () => file.click() } }),
      ta,
      busy ? h("button", { class: "send stop", title: "Stop (Esc)", text: "■", on: { click: stop } })
        : h("button", { class: "send", title: "Send (Enter)", text: "↑", on: { click: send } })),
    h("div", { class: "foot dim small" }, h("span", { text: settings.chat ? "chat template" : "raw completion" }), h("span", { text: "Enter sends · Shift+Enter newline · Esc stops" })));
  drawThumbs();
  function grow() { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 220) + "px"; }
  grow();
}
function drawThumbs() {
  const t = document.getElementById("thumbs");
  if (!t) return;
  t.replaceChildren(...pending.map((p, i) => h("div", { class: "thumb" }, h("img", { src: URL.createObjectURL(new Blob([p.data as BlobPart], { type: p.mediaType })), alt: "" }),
    h("button", { text: "×", title: "Remove", on: { click: () => { pending.splice(i, 1); drawThumbs(); } } }))));
}
async function attach(files: File[]) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    pending.push({ data: new Uint8Array(await f.arrayBuffer()), mediaType: f.type });
  }
  drawThumbs();
}
document.addEventListener("dragover", (e) => { if (e.dataTransfer?.types.includes("Files")) { e.preventDefault(); document.body.classList.add("drop"); } });
document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) document.body.classList.remove("drop"); });
document.addEventListener("drop", (e) => { document.body.classList.remove("drop"); if (e.dataTransfer?.files.length) { e.preventDefault(); attach([...e.dataTransfer.files]); } });

// ---- slash commands -------------------------------------------------------

interface Cmd { name: string; help: string; run(arg: string): void; }
const COMMANDS: Cmd[] = [
  { name: "/new", help: "start a new chat", run: () => switchTo(newChatNow()) },
  { name: "/delete", help: "delete this chat", run: () => removeChat(chat) },
  { name: "/chat", help: "apply the chat template", run: () => setChat(true) },
  { name: "/completion", help: "raw completion of the prompt", run: () => setChat(false) },
  { name: "/think", help: "show or hide thinking", run: toggleThink },
  { name: "/system", help: "set the system prompt: /system text", run: (a) => { settings.system = a; saveSettings(); toast(a ? "System prompt set" : "System prompt cleared"); } },
  { name: "/copy", help: "copy the last reply", run: copyLast },
  { name: "/panel", help: "show or hide the engine panel", run: togglePanel },
  { name: "/models", help: "open the model list", run: () => panel.showTab("models") },
  { name: "/help", help: "keyboard shortcuts", run: help },
];
function hints() {
  const box = document.getElementById("hints")!;
  const v = $<HTMLTextAreaElement>("prompt").value;
  if (!v.startsWith("/") || v.includes("\n")) { box.hidden = true; return; }
  const m = COMMANDS.filter((c) => c.name.startsWith(v.split(" ")[0]));
  box.hidden = m.length === 0;
  box.replaceChildren(...m.map((c) => h("button", { on: { click: () => { $<HTMLTextAreaElement>("prompt").value = c.name + " "; $("prompt").focus(); hints(); } } }, h("b", { text: c.name }), h("span", { class: "dim", text: " " + c.help }))));
}

// ---- generation -----------------------------------------------------------

function stop() { busy?.abort(); }

async function send() {
  const box = $<HTMLTextAreaElement>("prompt");
  const text = box.value.trim();
  if (text.startsWith("/")) {
    const [name, ...rest] = text.split(/\s+/);
    const c = COMMANDS.find((x) => x.name === name);
    box.value = ""; box.style.height = "auto"; hints();
    if (c) c.run(rest.join(" ")); else toast("Unknown command " + name, true);
    return;
  }
  if ((!text && pending.length === 0) || busy) return;
  if (!modelId()) { toast("Load a model first (Models tab)", true); panel.showTab("models"); return; }
  const user: Msg = { role: "user", content: text, pics: pending.length || undefined };
  if (pending.length) pictures.set(user, pending);
  pending = [];
  box.value = ""; box.style.height = "auto";
  chat.msgs.push(user);
  chat.model = modelId();
  touch(chat);
  drawLog();
  drawChats();
  drawThumbs();
  run();
}

async function run() {
  const model = modelId();
  const reply: Msg = { role: "assistant", content: "", model };
  chat.msgs.push(reply);
  const wrap = log.querySelector(".thread") ?? log.appendChild(h("div", { class: "thread" }));
  if (log.querySelector(".empty")) { log.replaceChildren(wrap); }
  const view = msgView(reply, chat.msgs.length - 1);
  wrap.append(view.root);
  stick = true; scrollEnd();

  busy = new AbortController();
  live.streaming = true;
  drawComposer(); drawHeader();
  const history = chat.msgs.slice(0, -1);
  const reqMsgs = history.flatMap((m) => {
    const [one] = requestMessages([m]);
    return one ? [{ ...one, images: (pictures.get(m) ?? []).map((p) => ({ data: p.data, mediaType: p.mediaType })) }] : [];
  });
  const lastUser = history[history.length - 1];
  const req = create(GenerateRequestSchema, {
    modelId: model,
    maxTokens: optNumber(settings.maxTokens) ?? 0,
    stop: parseStop(settings.stop),
    sampling: {
      temperature: optNumber(settings.temperature), topP: optNumber(settings.topP), topK: optNumber(settings.topK), minP: optNumber(settings.minP),
      repeatPenalty: optNumber(settings.repeatPenalty), presencePenalty: optNumber(settings.presence), frequencyPenalty: optNumber(settings.frequency),
      ...(optNumber(settings.seed) !== undefined ? { seed: BigInt(Math.trunc(optNumber(settings.seed)!)) } : {}),
    },
    prompt: { input: settings.chat
      ? { case: "chat", value: { addGenerationPrompt: true, messages: reqMsgs, ...(settings.system.trim() ? { system: settings.system.trim() } : {}) } }
      : { case: "text", value: lastUser.content } },
  });

  const t0 = performance.now();
  let first = 0, tokens = 0, sampleAt = t0, sampleTokens = 0;
  let paint = 0;
  const schedule = () => { if (!paint) paint = requestAnimationFrame(() => { paint = 0; view.update(); scrollEnd(); }); };
  const sampler = setInterval(() => {
    const now = performance.now();
    if (first && tokens - sampleTokens > 0) live.decode.push(((tokens - sampleTokens) * 1000) / (now - sampleAt));
    sampleAt = now; sampleTokens = tokens;
    changed();
  }, 500);
  const meta: Meta = {};
  try {
    for await (const ev of infer.generate(req, { signal: busy.signal })) {
      if (ev.event.case === "started") {
        live.last.prompt = ev.event.value.promptTokens;
        live.last.queued = Number(ev.event.value.queuedMillis);
      } else if (ev.event.case === "token") {
        if (!first) { first = performance.now(); meta.ttft = first - t0; live.last.ttft = meta.ttft; sampleAt = first; }
        tokens++;
        reply.content += ev.event.value.text;
        schedule();
      } else if (ev.event.case === "finished") {
        const f = ev.event.value;
        meta.reason = FinishReason[f.reason]?.toLowerCase().replace(/_/g, " ");
        meta.tokens = f.completionTokens;
        meta.tps = f.decodeTokensPerSecond || (Number(f.decodeMillis) > 0 ? (f.completionTokens * 1000) / Number(f.decodeMillis) : 0);
        meta.prompt = f.promptTokens; meta.cached = f.cachedTokens;
        meta.gbps = meta.tps && f.bytesPerToken ? (meta.tps * Number(f.bytesPerToken)) / 1e9 : 0;
        Object.assign(live.last, {
          tps: meta.tps, bytesPerToken: Number(f.bytesPerToken), gbps: meta.gbps, prompt: f.promptTokens, cached: f.cachedTokens, position: f.position,
          promptTps: Number(f.prefillMillis) > 0 ? ((f.promptTokens - f.cachedTokens) * 1000) / Number(f.prefillMillis) : 0,
        });
      }
    }
  } catch (e) {
    if (busy?.signal.aborted || isCancel(e)) meta.stopped = true;
    else { meta.error = errText(e); if (isUnauth(e)) live.needKey = true; }
  } finally {
    clearInterval(sampler);
    cancelAnimationFrame(paint);
    reply.meta = meta;
    busy = null;
    live.streaming = false;
    touch(chat);
    view.update();
    drawComposer(); drawHeader(); drawChats(); drawKey();
    scrollEnd();
    $("prompt").focus();
    changed();
  }
}

// ---- api key --------------------------------------------------------------

function drawKey() {
  const existing = document.getElementById("keybox");
  if (!live.needKey) { existing?.remove(); return; }
  if (existing) return;
  const input = h("input", { type: "password", placeholder: "API key", autocomplete: "off" });
  const box = h("form", { id: "keybox", class: "keybox", on: { submit: (e: Event) => { e.preventDefault(); keepKey(input.value || null); input.value = ""; live.needKey = false; box.remove(); refreshLists(); } } },
    h("b", { text: "This server asks for an API key" }), input, h("button", { class: "btn", text: "Use key" }));
  document.body.append(box);
  input.focus();
}

// ---- overlays: palette and help ----------------------------------------------

function overlay(content: Node) {
  const o = $("overlay");
  o.hidden = false;
  o.replaceChildren(h("div", { class: "sheet", on: { click: (e: Event) => e.stopPropagation() } }, content));
  o.onclick = closeOverlay;
}
function closeOverlay() { $("overlay").hidden = true; $("overlay").replaceChildren(); document.getElementById("prompt")?.focus(); }

function palette() {
  const acts: [string, string, () => void][] = [
    ["New chat", "alt+N", () => switchTo(newChatNow())],
    ["Delete this chat", "", () => removeChat(chat)],
    ["Copy last reply", "alt+Y", copyLast],
    ["Toggle thinking", "alt+T", toggleThink],
    ["Switch to chat template", "", () => setChat(true)],
    ["Switch to raw completion", "", () => setChat(false)],
    ["Toggle engine panel", "alt+B", togglePanel],
    ["Toggle chat list", "alt+L", toggleChats],
    ["Larger or smaller text", "", () => { settings.bigText = !settings.bigText; saveSettings(); layout(); }],
    ["Engine: throughput and blocks", "", () => { openPanel(); panel.showTab("engine"); }],
    ["Devices and memory", "", () => { openPanel(); panel.showTab("devices"); }],
    ["Models: load and unload", "", () => { openPanel(); panel.showTab("models"); }],
    ["Sampling and system prompt", "", () => { openPanel(); panel.showTab("settings"); }],
    ["Keyboard shortcuts", "?", help],
    ...loadedIds().map((id) => ({ modelId: id })).map((m) => ["Use model " + m.modelId, "", () => { settings.model = m.modelId; chat.model = m.modelId; saveSettings(); drawHeader(); changed(); }] as [string, string, () => void]),
    ...allChats().slice(0, 30).map((c) => ["Open chat: " + c.title, "", () => switchTo(c)] as [string, string, () => void]),
  ];
  const input = h("input", { class: "pin", placeholder: "Type a command…", autocomplete: "off" });
  const list = h("div", { class: "plist" });
  let sel = 0, shown = acts;
  const draw = () => {
    const q = input.value.toLowerCase();
    shown = acts.filter((a) => a[0].toLowerCase().includes(q));
    sel = Math.min(sel, Math.max(0, shown.length - 1));
    list.replaceChildren(...shown.map((a, i) => h("button", { class: i === sel ? "on" : "", on: { click: () => { closeOverlay(); a[2](); }, mousemove: () => { sel = i; draw(); } } }, h("span", { text: a[0] }), a[1] ? h("kbd", { text: a[1] }) : null)));
  };
  input.addEventListener("input", () => { sel = 0; draw(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(sel + 1, shown.length - 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); draw(); e.preventDefault(); }
    else if (e.key === "Enter") { const a = shown[sel]; if (a) { closeOverlay(); a[2](); } }
  });
  overlay(h("div", {}, input, list));
  draw();
  input.focus();
}
function openPanel() { if (!settings.panel) togglePanel(); else if (window.innerWidth < 1100) drawerState("panel"); }

function help() {
  const rows: [string, string][] = [["Enter", "send"], ["Shift+Enter", "new line"], ["Esc", "stop the reply, or close this"], ["Up (empty prompt)", "recall your last message"],
    ["Alt+N", "new chat"], ["Alt+Up / Alt+Down", "previous / next chat"], ["Alt+T", "show or hide thinking"], ["Alt+Y", "copy the last reply"],
    ["Alt+B", "engine panel"], ["Alt+L", "chat list"], ["Ctrl+K", "command palette"], ["/", "slash commands in the prompt"], ["?", "this help"]];
  overlay(h("div", { class: "helpbox" }, h("h2", { text: "Keyboard shortcuts" }), ...rows.map(([k, v]) => h("div", { class: "hrow" }, h("kbd", { text: k }), h("span", { text: v })))));
}

document.addEventListener("keydown", (e) => {
  const typing = document.activeElement instanceof HTMLInputElement || document.activeElement instanceof HTMLTextAreaElement || document.activeElement instanceof HTMLSelectElement;
  const k = e.key.toLowerCase();
  if (e.key === "Escape") {
    if (!$("overlay").hidden) { closeOverlay(); return; }
    if (open) { drawerState(""); return; }
    if (busy) { stop(); return; }
    return;
  }
  if ((e.ctrlKey || e.metaKey) && k === "k") { e.preventDefault(); palette(); return; }
  if (e.altKey && !e.ctrlKey && !e.metaKey) {
    if (k === "n") { e.preventDefault(); switchTo(newChatNow()); }
    else if (k === "t") { e.preventDefault(); toggleThink(); }
    else if (k === "y") { e.preventDefault(); copyLast(); }
    else if (k === "b") { e.preventDefault(); togglePanel(); }
    else if (k === "l") { e.preventDefault(); toggleChats(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); step(-1); }
    else if (e.key === "ArrowDown") { e.preventDefault(); step(1); }
  }
  if (e.key === "?" && !typing) { e.preventDefault(); help(); }
  else if (!typing && !e.ctrlKey && !e.metaKey && !e.altKey && e.key.length === 1) $("prompt")?.focus();
});

// ---- boot -----------------------------------------------------------------

function drawAll() {
  drawHeader(); drawChats(); drawLog(); drawComposer();
}
panel.mount($("panel"), { model: modelId, toast, setModel: (id) => { settings.model = id; chat.model = id; saveSettings(); drawHeader(); changed(); } });
document.addEventListener("jitllm-settings", () => { drawHeader(); layout(); });
layout();
drawAll();
let lastList = "";
subscribe(() => {
  // The header and the empty state follow the loaded list; the chat itself is left alone while a reply streams.
  const sig = JSON.stringify([loadedIds(), live.connected, live.info?.version, live.needKey]);
  if (sig !== lastList) {
    lastList = sig;
    drawHeader();
    drawKey();
    if (!busy && chat.msgs.length === 0) drawLog();
  } else {
    const chip = header.querySelector(".rate");
    const rate = live.streaming ? live.decode.last : live.last.tps;
    if (chip) { chip.textContent = rate ? `${rate.toFixed(1)} tok/s` : "idle"; chip.classList.toggle("live", live.streaming); }
  }
});
start();
void modelsApi; void hasKey; void human; void clear; void saveChats;
