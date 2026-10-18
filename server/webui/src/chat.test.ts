import { test } from "node:test";
import assert from "node:assert/strict";
import { parseStop, optNumber, requestMessages, rate, takeKey } from "./chat.ts";

test("stop strings drop blank lines", () => {
  assert.deepEqual(parseStop("a\n\n  b \n"), ["a", "b"]);
});
test("an empty number is unset, zero is zero", () => {
  assert.equal(optNumber(""), undefined);
  assert.equal(optNumber("0"), 0);
  assert.equal(optNumber("x"), undefined);
});
test("a stopped empty reply is not sent back", () => {
  const m = requestMessages([
    { role: "user", content: "hi" },
    { role: "assistant", content: "" },
    { role: "user", content: "again" },
  ]);
  assert.deepEqual(m.map((x) => x.role), ["user", "user"]);
});
test("the whole conversation is sent", () => {
  const m = requestMessages([
    { role: "user", content: "a" },
    { role: "assistant", content: "b" },
    { role: "user", content: "c" },
  ]);
  assert.equal(m.length, 3);
});
test("rate", () => {
  assert.equal(rate(100, 2000), "50.0 tok/s");
  assert.equal(rate(0, 10), "");
});
test("the key leaves the URL and keeps the other parameters", () => {
  assert.deepEqual(takeKey("?key=k&server=http%3A%2F%2Fh", null), { key: "k", cleaned: "?server=http%3A%2F%2Fh" });
  assert.deepEqual(takeKey("?key=k", null), { key: "k", cleaned: "" });
  assert.deepEqual(takeKey("", "s"), { key: "s", cleaned: null });
});
