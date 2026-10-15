# The kernel composer: fusion recipes chosen per card by timing

AGENTS.md names the standing goal: the JIT should compose kernels, choosing and
fusing from the kernels it already knows by the operation graph and the
hardware, rather than hand-writing each fusion into a call site. This note is
the representation that goal starts from and the first slice built on it.

## What exists, and why it is not yet a composer

The device tier already makes four per-card fusion decisions, each wired by
hand into `encodeLayer`'s launch sequence and each with its own switch:

| fusion | built by | decided by | evidence |
|---|---|---|---|
| operand prologue (norm/quantize inside the matvec), sites q/k/v, o, down | `fuseActIn` | `tuneActIn`: timed per shape key, cached `gpuactin` | GK "The prologue decided per card by measurement" |
| gate+up as one launch, down forming act(gate)*up (`gu`) | `fuseGateUp` | `Config.GateUpFuse`, off | GK "Gate and up in one launch: exact, and a loss past the 1B" |
| gate+up as one launch summing both rows (`Gate2`) | `gate2Kernel` | `tuneGateUp2`: Gate2 against two launches per up shape, cached `gpugateup2` | GK "... each workgroup summing both rows (Gate2)" |
| split / in-group split | `retuneDecode` | timed per shape, cached `gpusplit` | GK "split tuner" |

The FFN's three forms were a precedence chain (`GateUpFuse` before `Gate2`
before two launches): a boolean for `gu`, and a two-arm tuner for Gate2 that
could not see `gu` or a form composed with the norm. `gu` won on the 1B (1.006) and lost 5-9% past it; Gate2 won 1-3%
on all three. A different card, a different FFN width or a different format
can reorder them, and the chain cannot see that. The operand prologue is
already a per-card choice, but its tuner is specific to its two arms.

## The representation

A **site** is a place in a block's graph where more than one generated form
computes the same values: the q/k/v operand, the attention output's operand,
the FFN's gate/up/down chain, down's operand. A **recipe** is one form at a
site: which kernels it launches, in what order, over which buffers. A recipe
is data: a name, a build step (which kernels it needs; nil where the shape
cannot express it), and an arm (the launches it makes for one block, the way
`encodeLayer` makes them). The composer's state is, per site and per shape
key, the index of the chosen recipe.

    type recipe struct {
        name  string
        ok    func(l *layer) bool                    // built and expressible here
        arm   func(s backend.Session, l *layer) error // one block's launches
    }
    type site struct {
        name    string
        key     func(l *layer) string // shape key; "" = not offered
        recipes []recipe               // recipes[0] is the reference form
    }

The shape key holds what decides cost and nothing else: format, k, rows,
split and reduction form of every matvec in the site. Two blocks with the
same key share a verdict, which is what lets a 32-block model time 16 blocks
once. The verdict is cached per binary, device and key (`gpurecipe`), as the
split tuner's is.

### Correctness is per recipe, not per site

Every recipe is held to the reference within a measured band, and its gate has
a violation that must fail. Where a recipe is exact on a backend (Gate2 on
PTX, `gu` everywhere), its gate is bit equality with recipe 0 and a swap
violation; where it is not (Gate2 on SPIR-V, where the driver contracts sums
differently), it is not offered on that backend until its band gate exists.
Recipes do not have to agree with each other bit for bit; byte-identical
goldens guard only a refactor of one recipe (emitter-patterns.md).

### The search

For each site, for each key, every offered recipe is timed over up to 16
placed blocks with that key, interleaved within the process (ABAB.., the arm
order rotated per round), three repetitions a round, ten rounds with the first
discarded for first touch. Each recipe's ratio is the median of its per-round
ratios to recipe 0; a trial whose IQR/median exceeds the split tuner's gate
(`splitTuneIQR`) does not resolve and keeps recipe 0. A recipe is kept only
where it is at least 1% faster than every recipe before it in the ranking
(`recipeMargin`, the prologue tuner's margin), so a tie keeps the simpler form.

Sites run in dependency order: a site whose arms include another site's
launches is timed after that site settles, with its verdict in force. The
FFN site's arms include down, so the down-prologue verdict is decided first
and down runs in the form it ships in. This is a greedy, not a joint, search:
a joint search over the product of sites' recipes is exponential, and the
measured interactions (down's prologue against `gu`'s gated prologue) live
inside one site's arms where they are timed together.

### What a forced choice does

Every switch survives as a forcing option (`With...`): `GateUpFuse` forces
`gu` where it is built, `GateUp2=false` removes Gate2 from the offer,
`ActInSites` forces the prologue sites, `NoFuse` keeps recipe 0 everywhere,
and `JITLLM_GPU_TUNE=0` keeps the recipe the defaults name (the fitted path:
Gate2 where exact, the prologue nowhere). Counters (`Stats().RecipeTimed`,
`RecipeKept`, per-recipe run counts) let a gate assert the configuration it
claims was selected (RULE 10).

## The five principles

- **JIT**: every recipe's kernels are emitted per card; the composer chooses
  among generated kernels and emits nothing pregenerated.
- **No Go compute**: the search is timing; no token arithmetic moves to Go.
- **Relocation and paging**: a recipe is per block and keyed by shape, so a
  block moved to another card takes that card's verdict for its key; a block
  paged out is skipped by the trial (`l.pg != nil && !l.pg.in`), as the
  prologue tuner skips it.
- **No allocation**: the trial runs at tune time, outside a token; a token's
  launch reads an int per site from the layer.

## Candidates, ranked by what the profile leaves

Decode on the V100 runs matvecs at 60-90% of the read wall; launches removed
by fusion have repeatedly measured below noise (RoPE+KV) or lost (persistent
and dependency-counter blocks). The candidates below are ranked by what a
recipe can buy without adding traffic.

1. **FFN gate/up/down as a site** (two launches, Gate2, `gu`): built; replaces
   two hand-wired booleans. First slice.
2. **Operand prologue as sites** (q/k/v, o, down): already timed; becomes three
   sites of the same tuner. First slice.
3. **Gate2 with the FFN norm's prologue** (`gate2in`: one launch normalising
   and quantizing the residual, summing the gate's and up's rows and writing
   act(gate)*up). The norm launch and the gate launch both go. The prologue at
   gate and up lost before because two launches each redid the norm; under
   Gate2 one launch does. Both pieces exist (`kernels.ActIn`,
   `MatVecShape.Gate2`); the recipe is their composition, the first found by
   composing rather than written.
4. **Residual + next norm statistics in down's epilogue**: the next block's
   q/k/v prologue recomputes the row's sum of squares in every workgroup;
   the codex review notes the exact form needs the existing partial order,
   and that a block must remain independently runnable after relocation.
5. **Attention output with o's prologue** and **RoPE into the q/k matvec
   epilogue**: small on the measured profile; kept as candidates.
6. **Host-side region fusion** (CPU pool): the dispatch cost is 2.7 us a
   region; the same table shape applies with the pool's regions as recipes.
7. **Prefill GEMM epilogues**: another agent's work (the add-norm fold);
   not duplicated here.

## The first slice, as built

`jit/gpu/tier/recipe.go`. `timeRecipes` is the one trial: N arms
interleaved in one process, the order rotated each round, 20 paired rounds
after one discarded, each challenger's per-round ratio to the incumbent,
median and IQR gate. `tuneActIn` runs on it (its three sites, two recipes
each), and `tuneFFN` replaces `tuneGateUp2`: per FFN shape key it times
two, gate2, gate2in and gu (each built where the shape takes it) norm
through down, keeps the device's default (Gate2 where exact, else two)
unless a challenger is 1% faster, and writes the Gate2 verdict
`gateUp2On` reads. `Config.FFNForm` (`JITLLM_GPU_FFN`) forces a recipe;
`GateUpFuse` and `GateUp2Force` still force theirs. `mvLaunch` is the one
launch the tuners' arms make, matching `encodeLayer`'s workgroup width,
row tile and partial reduction.

The recipe found by composing: **gate2in**, Gate2 with ActIn's norm
prologue. Where the attention output's launch already added the residual,
the FFN's RMSNormQuantRows goes and gate and up read the residual row
themselves: two launches become one per block. It is bit-identical on PTX
(`backend.TestGate2InIsTheTwoLaunches`, 96 cases; `model.TestFFNRecipesMatchTwoLaunches`
with every block placed with and without the graph and a half seam;
violation `TestGate2InSeesASwap`). The tuner keeps it on both of
Llama-3.2-1B's FFN keys on the V100. Rows: GK "The FFN's recipes, timed per
card".
