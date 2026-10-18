# The server orchestrator: placement driven by the server itself

jitllm's blocks, their KV pages and their recurrent state move between the
host and any device at runtime (the relocation principle). A user can move
them (`PlacementService.RelocateBlocks`). This design is the server moving
them itself: making room for a conversation that needs a card, giving the
cards to the conversations that are generating, and taking them back from the
ones that went quiet. Every decision is one a caller can force or forbid.

## What exists

- Preemption (`server/preempt.go`): a per-model KV budget; a generate that
  would exceed it parks other sessions of the same model (their history home
  and into a store), lowest priority then least recently used, and the parked
  session resumes byte for byte. It moves history, never weights, and never
  across models.
- Plasticity in the engine: `tier.GPU.AddDevices` (a running model takes a
  card), `State.PlaceBlocks` / `ReleaseBlocks` (any block range, with its
  history), `State.MoveBlocks` (between devices), binary16 histories that
  travel converted.

## v1: promote on demand, demote the idle to make room

**Promote.** When a session starts a generate (where preemption admits its
KV), the orchestrator offers the session's host blocks to the model's
devices (`PlaceBlocks`), so a conversation that is talking runs on a card
whenever a card can hold it -- also after it was demoted, and for a model
loaded on the CPU when the server has a device.

**Make room.** When the card declines for lack of room, the orchestrator
demotes the least deserving sessions on that card -- not generating (their
lock is free), lowest priority, then idle longest, sessions of other models
first -- by bringing their blocks home with their history (`ReleaseBlocks`),
then offers again. A block's weights leave the card when no session of its
model holds it there (the tier's rule). It never demotes a running session,
and it stops when the promotion fits or no victim is left: then the session
runs where it is, as before (nothing is refused).

**Leave alone.** A session the caller placed explicitly (RelocateBlocks,
SetPlacement, or a load's `max_device_blocks`) is pinned: the orchestrator
neither promotes nor demotes it until the caller releases the pin
(`SetPlacement` with orchestrated placement). The engine's own per-session
relocation on a full card (SetRelocation) keeps working underneath.

**Visible.** Every orchestrated move is recorded -- session, model, blocks,
from, to, why ("promote: generating", "make room for s-…"), how long it took
-- in a bounded log the stats stream carries, and counted (promotions,
demotions, blocks moved, bytes moved). The web UI's Placement tab and the TUI
show it.

**Forcible.** `jitllmd -orchestrate=false` (and an Engine option) turns it
off; per session, pinning as above; per model, `max_device_blocks` caps it.
Defaults on: the server reaches the best placement without user tuning.

## Gates

- A session demoted to make room and promoted back produces the same greedy
  tokens as one that never moved (and its KV is not recomputed: cached tokens
  carried).
- Two models on one card that cannot hold both: the one generating gets the
  card, alternately, every turn; no refusal, no recomputed prompt.
- A pinned session is never moved; `-orchestrate=false` moves nothing.
- No leaks across many promote/demote cycles (device ledger and host frames
  back to baseline), and the five principles' gates with it on.

## Later (v2)

Rebalancing across cards as load shifts (MoveBlocks between devices),
predictive promotion (a session about to generate: queued, typing), host-RAM
pressure (paging weights out of host for idle models), and a cost model that
weighs a move's milliseconds against the tokens it buys.
