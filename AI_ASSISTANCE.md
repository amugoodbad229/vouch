# AI assistance

Disclosure of how this project was made, as required by the event rules. Written to be
checkable rather than reassuring: every claim here can be verified against the git history.

## The short version

The code in this repository was written by an AI coding assistant, working under the
direction of a human who made every decision about what to build and can explain and defend
each one. All of it was written on or after **Oct 15, 2026**, inside the event window.

## Who did what

**The human** chose the problem, the three tracks, the product's claim, the graph model, the
tool surface, the refusal rule and the approval rule; supplied the logo; chose the visual
direction; reviewed every screen and every answer; and directed and corrected the work
throughout, including rejecting one approach outright ("just make a project that works") and
redirecting the effort more than once.

**The AI assistant** (Claude, via a coding-agent harness) wrote the code in this repository,
read the FalkorDB source alongside the human, wrote the probe scripts used to verify engine
behaviour, and drafted the documentation in this repository and the design notes that
preceded it.

The human did not read every line as it was written. The human *did* run the product, look at
its output, and reject what was wrong — several of the fixes recorded in the commit log exist
because the human or the human's reading of a screenshot found a fault the assistant had
missed. The commit log is the record.

## What happened before the event started

Preparation only, all of it discarded:

- Reading the FalkorDB source, its documentation, and two other open-source projects.
- Comparing FalkorDB `v4.22.0` against `6.0.1` to choose which to pin.
- Sketching the graph model, the ticket list and the evaluation design on paper and in a
  private planning repository.
- Running short throwaway scripts — roughly ninety of them — to confirm that specific query
  shapes, index commands, algorithm calls and temporal functions behave as documented.

Those scripts were run and thrown away. Their *findings* are carried forward as prose: they
are why the README can state which algorithm calls return zero rows silently, why a
variable-length bound is written as a literal, and why the engine is pinned. No script, and
no line of code, from that period is in this repository. The private planning repository is
not this one and is not published.

## What is third-party

| Thing | Origin | Licence |
|---|---|---|
| FalkorDB | `falkordb/falkordb:v4.22.0` container image | see the FalkorDB repository |
| Cytoscape.js 3.33.1 | vendored at `web/vendor/cytoscape.min.js` | MIT |
| The logo | drawn by the human | the human's own work |

Nothing here is copied from another project. Where an approach was informed by public source
code — the tool-catalogue shape, the "no path found is a real answer" distinction — that
influence is a design decision recorded in the commit messages, not copied code.

## What uses a model at runtime

Separate from the question above: the product calls an LLM to choose which tool to run. That
model never writes a query. It picks from a closed set of typed tools, the backend owns and
validates every Cypher statement, and a sentence-level guard withholds any answer that names
something the tools did not return. The design exists so that the model's unreliability is
contained rather than trusted.

## If this stops being true

Then this file gets corrected rather than the statement. A disclosure that is only accurate
at the moment it was written is not a disclosure.
