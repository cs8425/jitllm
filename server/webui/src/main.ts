import { createClient, ConnectError, Code } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { create } from "@bufbuild/protobuf";
import { ModelService } from "./gen/jitllm/v1/model_pb";
import { DeviceService } from "./gen/jitllm/v1/device_pb";
import { InferenceService, GenerateRequestSchema } from "./gen/jitllm/v1/inference_pb";
import { FinishReason } from "./gen/jitllm/v1/common_pb";
import { Turn, parseStop, optNumber, requestMessages, rate, takeKey } from "./chat.ts";

// Same-origin by default; ?server=http://host:port points at another jitllmd
// (it must list this page's origin in -allow-origin). The API key is moved out
// of the address into sessionStorage, never localStorage.
const KEY = "jitllm.key";
function stored(): string | null {
  try { return sessionStorage.getItem(KEY); } catch { return null; }
}
function keep(k: string | null) {
  try { k === null ? sessionStorage.removeItem(KEY) : sessionStorage.setItem(KEY, k); } catch { /* private window: the key lasts this page load */ }
}
const taken = takeKey(location.search, stored());
let key = taken.key;
if (taken.cleaned !== null) {
  keep(key);
  history.replaceState(null, "", location.pathname + taken.cleaned + location.hash);
}
const server = new URLSearchParams(location.search).get("server");
const transport = createConnectTransport({
  baseUrl: server ?? location.origin,
  interceptors: [(next) => (req) => { if (key) req.header.set("Authorization", "Bearer " + key); return next(req); }],
});
const models = createClient(ModelService, transport);
const devices = createClient(DeviceService, transport);
const infer = createClient(InferenceService, transport);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// Every cell is built with textContent: names and model output are untrusted.
function el(tag: string, text = "", cls = ""): HTMLElement {
  const e = document.createElement(tag);
  e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function fill(table: HTMLElement, rows: HTMLElement[][], empty: string) {
  table.replaceChildren();
  if (rows.length === 0) rows = [[el("td", empty, "dim")]];
  for (const cells of rows) {
    const tr = document.createElement("tr");
    for (const c of cells) tr.append(c);
    table.append(tr);
  }
}
function actionCell(label: string, run: () => Promise<unknown>): HTMLElement {
  const td = document.createElement("td");
  const b = el("button", label) as HTMLButtonElement;
  b.addEventListener("click", async () => {
    b.disabled = true;
    try { await run(); } catch (e) { fail(e); }
    await refresh();
  });
  td.append(b);
  return td;
}

function fail(e: unknown) {
  const msg = e instanceof ConnectError ? e.rawMessage || Code[e.code] : (e as Error).message;
  $("status").textContent = msg;
  $("status").className = "err";
  if (e instanceof ConnectError && e.code === Code.Unauthenticated) $("login").hidden = false;
}

let lastSig = "";
let polling = false;
async function refresh() {
  // A slow server must not collect a queue of polls behind its first.
  if (polling) return;
  polling = true;
  try {
    const [m, d] = await Promise.all([models.listModels({}), devices.listDevices({})]);
    // Rebuilding the tables under the pointer swallows a click that lands
    // between two polls, so nothing is touched when nothing changed.
    const sig = JSON.stringify([m.loaded.map((x) => [x.modelId, x.blockCount]), m.files.map((f) => [f.path, f.modelId]), d.devices.map((x) => [x.ref?.id, x.freeMemory?.human])]);
    $("status").textContent = "connected";
    $("status").className = "dim";
    $("login").hidden = true;
    if (sig === lastSig) return;
    lastSig = sig;
    fill($("devices"), d.devices.map((x) => [
      el("td", x.ref?.id ?? ""), el("td", x.name), el("td", `${x.freeMemory?.human ?? ""} / ${x.totalMemory?.human ?? ""}`, "n"),
    ]), "no devices");
    fill($("loaded"), m.loaded.map((x) => [
      el("td", x.modelId), el("td", `${x.architecture}, ${x.blockCount} blocks, ctx ${x.contextLength}`, "n"),
      actionCell("Unload", () => models.unloadModel({ modelId: x.modelId })),
    ]), "nothing loaded");
    // A projector alone is not a model: it loads as part of the one it serves.
    const files = m.files.filter((f) => f.isContainer && f.modelId === "" && !f.name.startsWith("mmproj-"));
    fill($("files"), files.map((f) => [
      el("td", f.name), el("td", f.size?.human ?? "", "n"), actionCell("Load", () => models.loadModel({ path: f.path })),
    ]), "no containers in the model directory");
    const sel = $<HTMLSelectElement>("model");
    const prev = sel.value;
    sel.replaceChildren(...m.loaded.map((x) => new Option(x.modelId)));
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  } catch (e) {
    lastSig = "";
    fail(e);
  } finally {
    polling = false;
  }
}

const history_: Turn[] = [];
let pending: { data: Uint8Array; mediaType: string }[] = [];
let busy: AbortController | null = null;

function setBusy(on: boolean) {
  $<HTMLButtonElement>("send").disabled = on;
  $<HTMLButtonElement>("stop").disabled = !on;
}

function bubble(role: string, text: string, images: Turn["images"] = []): HTMLElement {
  const d = el("div", "", "msg " + role);
  for (const im of images ?? []) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(new Blob([im.data as BlobPart], { type: im.mediaType }));
    d.append(img);
  }
  d.append(document.createTextNode(text));
  $("log").append(d);
  return d;
}

async function send() {
  const box = $<HTMLTextAreaElement>("prompt");
  const text = box.value.trim();
  const model = $<HTMLSelectElement>("model").value;
  if ((!text && pending.length === 0) || busy) return;
  if (!model) { fail(new Error("load a model first")); return; }
  box.value = "";
  const turn: Turn = { role: "user", content: text, images: pending };
  pending = [];
  $("attached").textContent = "";
  history_.push(turn);
  bubble("user", text, turn.images);
  const reply: Turn = { role: "assistant", content: "" };
  history_.push(reply);
  const cur = bubble("bot", "");
  const body = document.createTextNode("");
  cur.append(body);
  const meta = el("span", "", "meta");
  cur.append(meta);

  busy = new AbortController();
  setBusy(true);
  const req = create(GenerateRequestSchema, {
    modelId: model,
    maxTokens: optNumber($<HTMLInputElement>("max_tokens").value) ?? 0,
    stop: parseStop($<HTMLTextAreaElement>("stop").value),
    sampling: {
      temperature: optNumber($<HTMLInputElement>("temperature").value),
      topP: optNumber($<HTMLInputElement>("top_p").value),
    },
    prompt: { input: { case: "chat", value: {
      addGenerationPrompt: true,
      messages: requestMessages(history_.slice(0, -1)),
    } } },
  });
  try {
    for await (const ev of infer.generate(req, { signal: busy.signal })) {
      if (ev.event.case === "token") {
        reply.content += ev.event.value.text;
        body.data = reply.content;
      } else if (ev.event.case === "finished") {
        const f = ev.event.value;
        meta.textContent = [FinishReason[f.reason], `${f.completionTokens} tokens`, rate(f.completionTokens, Number(f.decodeMillis))].filter(Boolean).join(" · ");
      }
    }
  } catch (e) {
    const stopped = busy.signal.aborted;
    meta.textContent = stopped ? "stopped" : (e instanceof ConnectError ? e.rawMessage || Code[e.code] : (e as Error).message);
    if (!stopped) meta.className = "meta err";
    if (e instanceof ConnectError && e.code === Code.Unauthenticated) $("login").hidden = false;
  } finally {
    busy = null;
    setBusy(false);
    $("log").scrollTop = $("log").scrollHeight;
  }
}

$("send").addEventListener("click", send);
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("stop").addEventListener("click", () => busy?.abort());
$("clear").addEventListener("click", () => { busy?.abort(); history_.length = 0; $("log").replaceChildren(); });
$("attach").addEventListener("click", () => $("file").click());
$<HTMLInputElement>("file").addEventListener("change", async (e) => {
  const input = e.target as HTMLInputElement;
  for (const f of Array.from(input.files ?? [])) {
    pending.push({ data: new Uint8Array(await f.arrayBuffer()), mediaType: f.type });
  }
  input.value = "";
  $("attached").textContent = pending.length ? `${pending.length} picture(s) attached` : "";
});
$("login").addEventListener("submit", (e) => {
  e.preventDefault();
  key = $<HTMLInputElement>("key").value || null;
  keep(key);
  $<HTMLInputElement>("key").value = "";
  refresh();
});
refresh();
setInterval(refresh, 5000);
