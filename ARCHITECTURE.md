# Prism — Architecture

> `pip install prismlabs` · command: `prism`

---

## Overview

Prism is a scientific research CLI that covers the full arc from observation to publication. It is structured around **Investigations** — each with discrete phases, a local SQLite store, and an orchestrator agent that calls specialized tools and models at each step.

The core intelligence pipeline is:

```
Semantic Scholar (N papers)
        │
        ▼
Thinker Model  ──── explores each reference paper → candidate ideas
        │
        ▼
Orchestrator Agent  ──── plans experiments, manages phase state
        │
        ▼
Judge Model  ──── scores candidate ideas against reference papers
        │
        ▼
Conclusion + optional public broadcast
```

---

## 1. Package

| Field | Value |
|---|---|
| PyPI name | `prismlabs` |
| Entry point | `prism` |
| Python | ≥ 3.11 |

The package ships no model weights. Models are accessed through a **backend** abstraction (see §4).

---

## 2. Data Layer

### 2.1 `Paper`

Replaces the setter-based Paper class with a proper Pydantic model. Semantic Scholar results are parsed directly into this schema.

```
Paper
  id                str          # Semantic Scholar paperId
  title             str
  abstract          str
  authors           list[str]
  year              int
  publication_date  str          # ISO 8601
  venue             str | None
  citation_count    int
  fields_of_study   list[str]
  url               str
  arxiv_id          str | None
  future_explorations str | None  # extracted by Thinker or from paper body
  tldr              str | None   # SS-provided TLDR where available
  references        list[str]    # paperId list of direct references
  is_open_access    bool
```

Semantic Scholar returns all of these fields natively. The `semanticscholar` PyPI package wraps the API; `Paper.from_ss(ss_result)` is a single classmethod that does the mapping.

### 2.2 `Experiment`

```
Experiment
  id                str          # uuid
  investigation_id  str
  title             str
  description       str
  hypothesis        str          # explicit hypothesis the experiment tests
  code              str | None   # generated or user-written
  config            dict         # isolation flags, dependencies, env vars
  status            Literal["pending","running","complete","failed","skipped"]
  result            str | None   # stdout + interpreted output
  interpretation    str | None   # AI narrative of what result means
  is_isolated       bool         # True = result does not affect sibling experiments
  created_at        str
  updated_at        str
```

The `config.is_isolated` flag controls whether the orchestrator can let a surprising result pivot the experimental plan. If `is_isolated=False`, the orchestrator re-evaluates the remaining plan after interpreting the result.

### 2.3 `Investigation`

```
Investigation
  id                str
  title             str
  topic             str
  phase             Literal["observation","research","question","experiment","conclusion"]
  visibility        Literal["private","public"]   # §6
  user_context      str | None   # synthesized background
  papers            list[Paper]
  thinker_ideas     list[ProposedIdea]  # raw output from Thinker explores
  research_question str | None
  experiments       list[Experiment]
  novelty_scores    list[NoveltyScore]  # Judge outputs
  conclusion        str | None
  bibtex            str | None
  created_at        str
  updated_at        str
```

### 2.4 `ProposedIdea`

```
ProposedIdea
  title             str
  abstract          str
  source_paper_id   str         # which reference paper spawned this
  thinking_trace    str         # Thinker's chain-of-thought (stored, not shown by default)
  selected          bool        # user confirmed this as the working idea
```

### 2.5 `NoveltyScore`

```
NoveltyScore
  idea_title        str
  reference_paper_id str
  winner            Literal["idea","paper"]
  judge_reasoning   str
  score             float       # 0.0–1.0 derived from Judge win-rate across N comparisons
```

---

## 3. Investigation Workflow

### Phase 1 — Observation

1. User provides a topic (and optionally their own early idea).
2. Semantic Scholar is queried with the topic string. The global config `observation.paper_limit` (default 10, user-editable) controls N.
3. Results are parsed into `Paper` objects and displayed interactively.
4. User selects which papers to include.
5. Phase advances to `research`.

### Phase 2 — Background Research

1. Each selected paper's keywords + fields are used to issue a second, deeper Semantic Scholar query.
2. AI synthesizes a background narrative across all papers, stored in `investigation.user_context`.
3. Phase advances to `question`.

### Phase 3 — Question + Ideation

This is the core novel step.

1. The orchestrator asks the user whether they already have a working idea, or want Thinker to generate one.
2. **Thinker ideation loop:**
   - For each reference paper, `Thinker.explore(paper, use_future_explorations=True)` is called.
   - If the paper has a "Future Work" section (extracted during ingestion), `future_explorations` is populated, enabling the richer prompt path.
   - Each call returns `(thinking_trace, proposed_ideas)`. The thinking trace is stored but not rendered by default — users can `prism idea trace <id>` to inspect it.
   - All proposals are collected as `ProposedIdea` objects.
3. The orchestrator clusters similar ideas, deduplicates near-duplicates, and presents the shortlist to the user.
4. User selects or types their own. The chosen idea becomes `investigation.research_question`.
5. Phase advances to `experiment`.

### Phase 4 — Experimentation

1. Orchestrator generates an experiment plan (3–5 experiments) aligned to the research question and background.
2. Each experiment has an explicit `hypothesis` field and an `is_isolated` flag.
3. For each experiment:
   - User can `[r]un / [e]dit / [s]kip / [v]iew`.
   - Running executes the code in a subprocess (see §5 — Orchestrator Tools).
   - After execution, AI interprets the result.
   - If `is_isolated=False` and the result is surprising, the orchestrator re-plans remaining experiments.
4. Results are stored in `Experiment.result` and `Experiment.interpretation`.
5. Phase advances to `conclusion`.

> **Note:** In the current version, code execution is simulated (subprocess runs generated Python). Real simulation integrations (MATLAB, Isaac Sim, etc.) are roadmap items exposed as pluggable orchestrator tools.

### Phase 5 — Conclusion + Novelty Scoring

1. **Judge scoring:** For each `ProposedIdea` that was shortlisted (or just the selected one), the Judge compares it against each reference paper and produces a `NoveltyScore`. Win-rate across N comparisons is the novelty signal.
2. AI generates the full paper (Abstract → References) grounded in experiment results and novelty scores.
3. Output is saved to `~/.prism/investigations/<id>/paper.md` and `bibtex.bib`.
4. If `investigation.visibility == "public"`, the abstract + novelty metadata are broadcast to the Prism registry (§6).

---

## 4. Model Backend — No `from_pretrained` in CLI

The Thinker and Judge models are 30B parameter models. Loading them inline in a CLI process is wrong for several reasons: startup latency, memory management, and UX. The CLI never touches `transformers` directly.

### Backend Abstraction

```
prism/backends/
  base.py          # ModelBackend ABC: explore(), judge()
  local.py         # Runs a local vllm or llama.cpp server, communicates via OpenAI-compat HTTP
  remote.py        # Points to a hosted endpoint (HF Inference Endpoints, replicate, etc.)
  mock.py          # Returns fixture data; used in tests and when no backend is configured
```

The CLI calls `backend.explore(paper)` and `backend.judge(paper_a, paper_b)`. The backend is selected at `prism init` time and stored in config. The `local` backend launches a vllm subprocess on first use and keeps it running as a daemon (`prism serve start/stop/status`).

### `prism serve`

```bash
prism serve start           # boots vllm with SciThinker-30B + SciJudge-30B
prism serve start --thinker-only
prism serve status          # shows model load state, memory usage
prism serve stop
```

The serve daemon writes a socket/port to `~/.prism/serve.lock`. The CLI checks this at startup. If the daemon is not running and the backend is `local`, the CLI prompts the user to run `prism serve start`.

For users who cannot run models locally, `remote` backend accepts any OpenAI-compatible base URL:

```bash
prism config set backend.type remote
prism config set backend.base_url https://api-inference.huggingface.co/models/OpenMOSS-Team/SciThinker-30B
prism config set backend.api_key hf_...
```

---

## 5. Orchestrator Agent

The orchestrator is the single stateful agent that drives phase transitions and calls tools. It is powered by the user's configured OpenAI model (default `gpt-4o`).

### Tool Registry

```
tools/
  semantic_scholar.py   search_papers(query, limit), get_paper(id), get_citations(id)
  thinker.py            explore_paper(paper) → list[ProposedIdea]
  judge.py              compare(idea, reference_paper) → NoveltyScore
  executor.py           run_python(code, timeout) → ExecResult
  file_io.py            read_file(path), write_file(path, content)
  web_search.py         search(query) → list[SearchResult]   [future]
  lab_interface.py      [stub] connect to physical equipment  [future]
```

The orchestrator decides which tools to call based on phase and user messages. Tool results are fed back into the conversation context. All tool calls are logged to the investigation's event log, giving full replay capability.

---

## 6. Public / Private Investigations

Each investigation has a `visibility` field, set at creation time and changeable with `prism investigate visibility <public|private>`.

### Private (default)

Investigation data stays local. Nothing leaves the machine except:
- Semantic Scholar API calls (topic query only, no user data)
- OpenAI API calls (abstracts and context — user opts in at `prism init`)

### Public

When an investigation is public, Prism sends the following to the **Prism Registry** (a lightweight hosted index):

```json
{
  "investigation_id": "...",
  "abstract_preview": "...",      // first 200 chars of research question
  "fields_of_study": [...],
  "novelty_scores": [...],        // aggregate score only, no raw judge output
  "author_handle": "...",         // optional; set in prism init
  "timestamp": "..."
}
```

The registry enables:

1. **Collision detection:** When you define your research question, the orchestrator queries the registry and surfaces anyone publicly building in the same direction. Score similarity is computed over abstract embeddings. Output: *"2 other researchers are exploring adjacent ideas — want to compare notes?"*

2. **Open collaboration prompts:** If similarity is above a threshold, Prism can surface a contact (if the other user has enabled it) or just show the overlap score + their public abstract so you can decide whether to differentiate or collaborate.

3. **Community novelty signal:** Aggregate public novelty scores inform a leaderboard of the most novel open investigations in each field — a kind of community-driven arXiv trending.

> The registry is opt-in, read-write only for the user's own investigations, and publicly readable in aggregate only. No raw experimental data is ever transmitted.

---

## 7. Global Config Schema

Stored at `~/.prism/config.json`. Fully editable via `prism config set <key> <value>` or directly.

```jsonc
{
  "openai_api_key": "sk-...",
  "model": "gpt-4o",

  "user_profile": {
    "name": "Tarush Gupta",
    "handle": "@tarush",             // used for public registry
    "research_areas": ["AI interpretability", "post-training"],
    "allow_contact": true            // enables collaboration pings
  },

  "observation": {
    "paper_limit": 10,               // N papers from Semantic Scholar
    "fields_filter": [],             // e.g. ["Computer Science"] to narrow
    "include_tldr": true
  },

  "backend": {
    "type": "local",                 // "local" | "remote" | "mock"
    "base_url": "http://localhost:8000/v1",
    "api_key": "",
    "thinker_model": "OpenMOSS-Team/SciThinker-30B",
    "judge_model": "OpenMOSS-Team/SciJudge-30B"
  },

  "thinker": {
    "num_ideas_per_paper": 3,        // config.num_new_ideas equivalent
    "use_future_explorations": true  // enables the richer explore prompt
  },

  "judge": {
    "comparisons_per_idea": 5,       // how many reference papers to compare against
    "novelty_threshold": 0.6         // min win-rate to consider idea novel
  },

  "registry": {
    "enabled": false,                // public/private default
    "endpoint": "https://registry.prismlabs.ai"
  },

  "active_investigation_id": null
}
```

---

## 8. CLI Command Surface

```
prism init                                  configure API key, profile, backend
prism serve start|stop|status               manage local model daemon

prism investigate new <topic> [--public]    create investigation
prism investigate list
prism investigate use <id|index>
prism investigate status
prism investigate visibility <public|private>

prism observe [--limit N]                   Phase 1: Semantic Scholar discovery
prism research                              Phase 2: deep synthesis
prism question                              Phase 3: question + Thinker ideation
prism idea list                             view all Thinker-proposed ideas
prism idea trace <id>                       inspect Thinker chain-of-thought
prism idea select <id>                      manually select an idea

prism experiment plan                       Phase 4: generate experiment plan
prism experiment list
prism experiment run <id>
prism experiment log <id>                   view result + interpretation

prism judge                                 run Judge scoring on selected idea
prism conclude                              Phase 5: generate paper + novelty report

prism go                                    autopilot all remaining phases
prism chat                                  free-form AI chat within investigation context

prism config show
prism config set <key> <value>
prism config get <key>
```

---

## 9. Known Issues in Source (`AI-Can-Learn-Scientific-Taste/models.py`)

These are fixed in the Prism integration layer before any models are called:

| Issue | Location | Fix |
|---|---|---|
| `adjacentPaper.abstract` hardcoded to `userPaper.abstract` | `Judge.assessment()` line 30 | Corrected in `backends/local.py` judge prompt builder |
| `explore(currexplore=False)` still uses `user_prompt_currexplore` | `Thinker.explore()` line 120 | Fixed in `backends/local.py` thinker prompt builder — `use_future_explorations` flag drives correct branch |
| `return Tuple(...)` instead of `return (...)` | `Thinker.explore()` line 148 | Wrapped in `backends/local.py`; return is unpacked and typed properly |

---

## 10. Roadmap

| Phase | Feature |
|---|---|
| v0.2 | `prism serve` daemon + vllm backend |
| v0.2 | Semantic Scholar integration replacing arXiv |
| v0.2 | Thinker ideation loop in Phase 3 |
| v0.2 | Judge novelty scoring in Phase 5 |
| v0.3 | Prism Registry (public investigations + collision detection) |
| v0.3 | `prism idea trace` for Thinker chain-of-thought inspection |
| v0.4 | Non-isolated experiment pivoting (dynamic plan rewrite) |
| v0.4 | Remote backend support (HF Inference Endpoints, replicate) |
| v1.0 | Lab equipment integration stubs |
| v1.0 | arXiv direct submission from `prism conclude` |
