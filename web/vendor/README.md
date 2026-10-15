# Vendored front-end dependencies

Not installed by `uv sync` — these are browser assets, so they are copied in and
committed. Anything here is third-party and keeps its own licence.

| File | Version | Source | Licence |
|---|---|---|---|
| `cytoscape.min.js` | 3.33.1 | `https://cdn.jsdelivr.net/npm/cytoscape@3.33.1/dist/cytoscape.min.js` | MIT (© 2016–2025 The Cytoscape Consortium) |

Why vendored rather than loaded from a CDN: the demo has to run on whatever
network the reviewer has, and a graph that fails to draw because a CDN was slow
is a failure with nothing to do with this project. The licence text is in the
file's own header comment.

To refresh:

```sh
curl -sL -o web/vendor/cytoscape.min.js \
  https://cdn.jsdelivr.net/npm/cytoscape@3.33.1/dist/cytoscape.min.js
```
