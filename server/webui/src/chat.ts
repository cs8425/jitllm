// Pure helpers behind the chat view, kept free of the DOM and the generated
// client so node's test runner can run them.

export interface Turn {
  role: "user" | "assistant";
  content: string;
  // Media types of the pictures attached to a user turn, for display; the
  // bytes live beside the turn in the page.
  images?: { data: Uint8Array; mediaType: string }[];
}

// Stop strings are typed one per line; blank lines mean nothing.
export function parseStop(text: string): string[] {
  return text.split("\n").map((s) => s.trim()).filter((s) => s !== "");
}

// A number field left empty is "unset" (the server's default), not zero:
// temperature 0 is greedy and an absent one is the engine's own default.
export function optNumber(text: string): number | undefined {
  const t = text.trim();
  if (t === "") return undefined;
  const n = Number(t);
  return Number.isFinite(n) ? n : undefined;
}

// The messages a request carries: the whole conversation, since the UI sends
// every turn as a stateless chat and the server re-prefills (or answers from
// its prefix cache). An assistant turn that was stopped before any token came
// back is dropped so the next request does not carry an empty reply.
export function requestMessages(history: Turn[]) {
  return history
    .filter((t) => t.role === "user" || t.content !== "")
    .map((t) => ({ role: t.role, content: t.content, images: t.images ?? [] }));
}

export function rate(tokens: number, millis: number): string {
  if (millis <= 0 || tokens <= 0) return "";
  return (tokens / (millis / 1000)).toFixed(1) + " tok/s";
}

// The key travels in the URL only to be moved out of it: it is read once,
// kept in sessionStorage (never localStorage) and the address bar is cleaned.
export function takeKey(search: string, stored: string | null): { key: string | null; cleaned: string | null } {
  const q = new URLSearchParams(search);
  const k = q.get("key");
  if (k === null) return { key: stored, cleaned: null };
  q.delete("key");
  const rest = q.toString();
  return { key: k, cleaned: rest === "" ? "" : "?" + rest };
}
