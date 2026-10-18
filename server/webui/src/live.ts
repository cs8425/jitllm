// The engine as the page sees it: polled lists, the stats stream and the rates
// measured on this page's own generations. Subscribers are told on each change.
import type { ListModelsResponse } from "./gen/jitllm/v1/model_pb";
import type { Device } from "./gen/jitllm/v1/device_pb";
import type { GetStatsResponse, ServerInfo } from "./gen/jitllm/v1/telemetry_pb";
import type { Placement } from "./gen/jitllm/v1/placement_pb";
import type { Session } from "./gen/jitllm/v1/session_pb";
import { models, devices, telemetry, placement, sessions, errText, isUnauth, isCancel } from "./api.ts";

export class Series {
  v: number[] = [];
  constructor(readonly cap = 120) {}
  push(x: number) { this.v.push(x); if (this.v.length > this.cap) this.v.shift(); }
  get last() { return this.v.length ? this.v[this.v.length - 1] : 0; }
  get peak() { return this.v.reduce((a, b) => Math.max(a, b), 0); }
  get mean() { return this.v.length ? this.v.reduce((a, b) => a + b, 0) / this.v.length : 0; }
}

export const live = {
  connected: false,
  error: "",
  needKey: false,
  list: null as ListModelsResponse | null,
  devs: [] as Device[],
  stats: null as GetStatsResponse | null,
  info: null as ServerInfo | null,
  place: null as Placement | null,
  session: null as Session | null,
  decode: new Series(),
  // The last finished generation, as this page measured and the server reported it.
  last: { tps: 0, promptTps: 0, ttft: 0, bytesPerToken: 0, gbps: 0, prompt: 0, cached: 0, queued: 0, position: 0 },
  streaming: false,
};

// The loaded models' ids, from the full list when it has arrived and from the
// stats stream before: listing walks the model folder and can be slow on a busy box.
export function loadedIds(): string[] {
  return live.list ? live.list.loaded.map((m) => m.modelId) : live.stats?.models.map((m) => m.modelId) ?? [];
}

const subs = new Set<() => void>();
export const subscribe = (f: () => void) => { subs.add(f); return () => subs.delete(f); };
let queued = false;
export function changed() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; subs.forEach((f) => f()); });
}

function fail(e: unknown) {
  live.connected = false;
  live.error = errText(e);
  if (isUnauth(e)) live.needKey = true;
  changed();
}

let polling = false;
export async function refreshLists() {
  if (polling) return; // a slow server must not collect a queue of polls
  polling = true;
  try {
    const [m, d] = await Promise.all([models.listModels({}), devices.listDevices({})]);
    live.list = m; live.devs = d.devices; live.connected = true; live.error = ""; live.needKey = false;
    changed();
  } catch (e) { fail(e); } finally { polling = false; }
}

async function watch() {
  for (;;) {
    try {
      for await (const s of telemetry.watchStats({ intervalMillis: 1000 })) {
        live.stats = s;
        if (live.list && s.models.map((m) => m.modelId).join() !== live.list.loaded.map((m) => m.modelId).join()) refreshLists();
        if (s.devices.length) live.devs = s.devices;
        live.connected = true;
        changed();
      }
    } catch (e) { if (!isCancel(e)) fail(e); }
    await new Promise((r) => setTimeout(r, 3000));
  }
}

export async function refreshPlacement(modelId: string) {
  try {
    const ss = (await sessions.listSessions({})).sessions.filter((s) => s.modelId === modelId);
    live.session = ss[0] ?? null;
    live.place = live.session ? (await placement.getPlacement({ sessionId: live.session.sessionId })).placement ?? null : null;
    changed();
  } catch { /* the Engine tab shows no map until a session exists */ }
}

export async function start() {
  refreshLists();
  setInterval(refreshLists, 15000);
  watch();
  try { live.info = (await telemetry.getServerInfo({})).info ?? null; changed(); } catch { /* shown when the server answers */ }
}
