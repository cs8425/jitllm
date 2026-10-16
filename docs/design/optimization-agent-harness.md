# Optimization implementation-agent harness

Task and acceptance template. This document specifies the proposed workflow and evidence artifacts; it is not an implemented runner or a replacement for AGENTS.md. Binding project rules remain in AGENTS.md. The research roadmap is in [external-optimization-ideas.md](external-optimization-ideas.md).

## Objective: optimize a complete plan

Minimize TTFT and maximize generated-token rate for a declared workload while preserving generated compute, relocation, paging and warm allocation requirements. Reduce the cost of adding optimizations and model compositions. A candidate is a complete execution plan, possibly containing several interacting levers. An isolated lever's regression is not sufficient reason to discard every combination containing it. A selected complete plan must meet the project's regression and correctness gates.

For example, compression can add decode instructions; alone it may lose. Fused decode can remove expanded memory traffic, and compressed residency can prevent evictions. The interaction must be measured. Likewise a partition can introduce partial-output traffic that a matching fusion eliminates. These are hypotheses, not established wins.

## Optimization stages covered by the harness

Every lever declares its stage: **offline** (conversion), **emit** (runtime compilation), or **continuous** (autotuning/profile-guided replanning and re-emission). A complete plan can combine all three. Conversion produces verified graphs and encoded data; emission generates device-specific compute; continuous optimization uses counters and controlled performance evidence to revise that compute as conditions change. Treat continuous optimization as a first-class implementation target, not merely fixed-kernel dispatch.

For a continuous lever, record its observation counters and sampling cost, trigger and profile scope, candidate construction, qualification procedure, re-emission/invalidation dependencies, safe activation point, hysteresis, rollback and preparation budget. Test a controlled workload change that triggers adaptation, assert the new emitted/selected variant executes, and test stable conditions that must not cause repeated recompilation. Compare adaptation enabled and disabled on the same workload trace with the same starting state and cache conditions, including observation, trial, compilation and recapture time. Preserve the existing measurement and allocation gates. A faster post-adaptation steady state alone does not establish that adaptation paid for itself.

### Persistent evidence and cross-instance acceptance

Continuous levers also declare a persistent evidence key, per-block/device binding, invalidation rules, negative-result scope and experiment claim/lease protocol. Persist qualified wins and clear measured losses; noisy or incomplete trials remain unresolved. A loss for A alone must not suppress A+B without evidence for that combination. Persist recipes and evidence, not executable kernels.

Required targeted cases for this subsystem: restart reuses a qualified verdict; two instances requesting the same trial produce one owner and share its published result; owner failure releases/expires the claim; a relocated block uses destination-qualified evidence or a legal generated default; incompatible versions/layouts/workloads reject stale verdicts; corrupted records do not prevent correct execution; an isolated losing lever remains eligible in an untested legal bundle. Demonstrate equivalent-block reuse only with an explicit equivalence key, and demonstrate that a materially different block does not collide. Distinguish cache-supported deduplication from fresh measurements required for new performance claims. Record cache/claim activity to prove reuse rather than merely inferring it from timing.

### Optional profile-guided offline maintenance

If this later feature is implemented, its task contract declares profile provenance and sufficiency, the offline candidate, approval or configured maintenance authority, scheduling/resource budgets, publication/rollback and evidence invalidation. Test foreground demand returning during preparation, insufficient temporary disk, process interruption, duplicate maintenance requests, mixed profile classes and existing sessions retaining an old artifact. Failed validation must leave the active artifact usable. A candidate does not become a winner merely because it builds or compresses more: measure its scoped whole-request effect and preparation cost. No scheduler is implied to exist by this specification.

## 1. Freeze a task contract before implementation

Fill out this record with concrete values; unknowns stay explicitly unknown:

```yaml
task:
  id: REQUIRED
  base_commit: REQUIRED
  objective: REQUIRED
  workload:
    model_and_artifact_hashes: REQUIRED
    input_trace_and_seed: REQUIRED
    prompt_and_generation_lengths: REQUIRED
    concurrency_and_context_classes: REQUIRED
    memory_budgets_and_relocation_trace: REQUIRED
    hardware_and_backend: REQUIRED
  metrics:
    primary: REQUIRED
    constraints: REQUIRED
    cold_and_warm_boundaries: REQUIRED
  scope:
    allowed_subsystems: REQUIRED
    excluded_work: REQUIRED
    required_reuse_demonstration: REQUIRED
  control:
    baseline_commit_and_configuration: REQUIRED
    generated_reference_plan: REQUIRED
  levers:
    - id: REQUIRED
      stage: offline_or_emit_or_continuous
      graph_pattern_and_guards: REQUIRED
      prerequisites: REQUIRED
      conflicts: REQUIRED
      expected_interactions: REQUIRED
      numerical_contract: REQUIRED
      affected_state_and_artifacts: REQUIRED
      observation_trigger_and_switch_contract: REQUIRED_FOR_CONTINUOUS
      persistent_evidence_key_and_invalidation: REQUIRED_FOR_CONTINUOUS
      block_device_binding_and_trial_ownership: REQUIRED_FOR_CONTINUOUS
  validation:
    independent_reference: REQUIRED
    required_checks_from_scripts_checks: REQUIRED
    forced_path_evidence: REQUIRED
    negative_cases: REQUIRED
  search:
    initial_candidate_plans: REQUIRED
    hardware_trial_budget: REQUIRED
    expansion_and_stop_conditions: REQUIRED
```

Use declared metric boundaries, not an arbitrary score that hides a TTFT regression behind decode throughput. If objectives conflict, report the Pareto alternatives and the tradeoff; do not silently redefine success. Scope expansion is an explicit task-contract revision. Architecture/format admission and required gates remain governed by AGENTS.md.

## 2. Ordered stages and required artifacts

| Stage | Work product | Evidence needed to advance |
|---|---|---|
| Inventory | Current-source map, relevant history, existing mechanisms, task contract | Confirm the starting revision and explain what is actually missing; no duplicate executor or tuner by assumption. |
| Semantics | Pattern and effect contract, logical/physical representation separation | Shape/layout/state guards and independent fixtures; unknown semantics are rejected. |
| Candidate construction | Explicit baseline and complete alternative plans | Every lever is identifiable; prerequisites, conflicts, pass order and dependent invalidations are recorded. |
| Local validation | Verifier, reference and round-trip results | Positive matches and near-miss counterexamples; tests demonstrate required behavior, not merely repeat implementation formulas. |
| Target correctness | Forced generated paths on required devices | Path identifiers/counters establish that the candidate ran; state, paging, relocation and allocations meet their gates. |
| Plan experiment | Raw paired measurements and candidate manifest | A/A control and required same-pass comparisons; complete work, preparation and memory included. |
| Interaction analysis | Ablations and explanation of gains/costs | Show whether the chosen combination requires its levers and whether order matters; qualify unavailable ablations. |
| Integration | Selected guarded plan, generated control, updated test mapping | Whole-request evidence, required coverage, second eligible graph where applicable, explicit unresolved limitations. |

Keep separate status for implementation, correctness, performance and integration. An implemented candidate with missing hardware is not validated support. Do not promote a source review or a passing compiler test into device correctness.

## 3. Search combinations without assuming monotonic improvement

1. Derive a small candidate set from the bottleneck and dependency graph. Record each plan's ordered transformations, recipes, layouts and guards. Equivalent plans can be deduplicated only when cost-relevant behavior is equivalent.
2. Test the control and the hypothesized complete combination. For two independent levers A and B, use control, A, B and A+B where all are legal. If B requires A, mark B-alone invalid and test control, A, A+B; do not manufacture an invalid ablation.
3. Retain enabling candidates even when they are locally slower. For larger sets, use a bounded search frontier that includes promising compatible bundles and their required prefixes, not only individually winning levers. Record unevaluated combinations; a bounded search does not establish a global optimum.
4. Evaluate pass ordering when one rewrite exposes another or changes layouts/resources. Invalidate stale legality, cost, capture and tuning decisions after relevant changes.
5. Use deterministic work/traffic counts and conservative resource estimates to order trials. Static infeasibility can reject a plan; a speculative cost estimate alone cannot prove a legal combination regresses. Increase search only to answer a named remaining interaction or uncertainty.
6. Select by complete-plan, whole-request evidence. Report A+B versus control and legal ablations using fresh controls under the project measurement rules. Do not multiply standalone speedups to predict a combination.
7. Leave losing or unverified combinations unselected. An intermediate candidate can regress while being useful to a successful combination; the shipping choice is the complete plan with guards, not an unconditional enabling change imposed on every workload.

For a larger winning combination, remove one lever at a time where legal to identify redundant or essential components. Ablation order can itself matter; do not claim unique attribution from a single removal. Correctness and the five principles apply to every candidate executed, even exploratory ones. A performance goal never authorizes incorrect arithmetic or broken paging.

## 4. Evidence that prevents shortcuts

| Possible shortcut | Required evidence or countercheck |
|---|---|
| Test passes because the new path never ran | Force the candidate, assert its recipe/path identifier and meaningful execution counts; include a deliberate violation that the gate detects. |
| Optimized code compared with itself | Independent oracle/reference values and randomized weights AND buffers, including nonzero positions and state transitions. |
| Family-specific workaround disguised as generic support | Match operations and semantic guards; demonstrate another eligible graph and an ineligible near-match. Review new family/name branches. |
| Better headline from changed model, precision, work or budget | Record artifact hashes, numerical contract, token counts, actual budgets, memory ownership and workload equivalence. Any intentional difference is a separately labeled experiment. |
| Kernel win presented as engine win | Whole-request TTFT/tok/s plus launch, transfer, scratch and preparation costs; keep kernel-only conclusions labeled. |
| Hidden expanded copy or stale state | Account for physical compressed/expanded bytes, scratch, metadata and transition peaks; compare live KV/recurrent state and continuation values. |
| Tuning mutates the measured session | Isolated/restored state, comparable residency and recorded trial setup; no uncharged preparation in one arm. |
| Missing devices/models treated as success | Preserve explicit missing/skipped status and exact commands; no deletion, weakened tolerance, skip switch or gate removal to obtain green. |
| Historical or noisy baseline creates a win | Fresh paired/interleaved control, A/A test, required repeats and dispersion gates; publish raw results and rejected-run reasons. |
| A single lever loses so the idea is dismissed | Test the declared legal bundle and its interactions before rejecting that bundle; record incomplete exploration honestly. |
| A favorable combination is forced everywhere | Guard its applicability by graph, device, shape, layout and residency evidence; retain generated alternatives. |

The task record, results and test gates are reviewable artifacts, not proof against a malicious author. Review changes to the evaluator, reference, tolerances and measurement harness separately from candidate changes. Do not let a candidate silently redefine its acceptance checks. An executable artifact validator can check completeness and consistency; independent review remains necessary for semantic validity.

## 5. Hardware sessions and stopping

Use existing `scripts/checks` and measurement tools. Prepare candidates and small fixtures before a target session; execute measurements serially. On the developer machine every heavy command uses the required wrapper, for example:

```sh
./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./... -count=1
```

This illustrates the wrapper, not an instruction to run the entire suite at every stage. Use the mapped targeted checks and the required larger cap where applicable. Benchmark hosts follow AGENTS.md's separate measurement rules; the laptop is not a performance scoreboard host.

Stop dependent performance work on correctness failure. Stop a noisy session rather than reporting an unresolved timing effect as neutral. When the trial budget is exhausted, preserve best qualified plans and unresolved candidates; do not claim optimality or invent a result. Stop optional testing once the named risk is resolved and mandatory coverage is satisfied. Missing hardware keeps affected acceptance items open while independent work proceeds.

## 6. Final handoff format

Report:

1. Task/base/final revisions and changed paths, including any dirty patch identity.
2. Implemented semantics, patterns, levers and exact candidate plans; what remains unimplemented.
3. Check matrix: passed, failed, skipped/missing, not run, with raw artifact paths and executed variant evidence.
4. Complete-plan and legal-ablation results, measurement boundaries, controls, variability and unresolved combinations.
5. Time, memory, traffic/locality, execution resources, coordination and correctness: improved, regressed, unchanged by construction or unresolved, with evidence.
6. Reuse demonstration and engineering cost: new runtime family branches, code touched, target-device time and optimizations reused.
7. Selection/integration verdict and remaining gates. No implemented/validated/fast/shipping claims beyond the evidence.

The first implementation slice for this harness is to encode the task manifest and result ledger around existing runners, and add a completeness checker that refuses missing required evidence or skipped gates. This document does not claim that checker exists yet. Keep optimization code independent of artifact reporting so measurement instrumentation does not become production token-path work.
