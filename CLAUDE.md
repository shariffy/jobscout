## graphify (optional)

Graphify is optional maintainer tooling. Forks do not need it to adapt or run JobScout.

If `graphify-out/graph.json` exists (built locally with graphify):
- For codebase questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, and `graphify explain "<concept>"` before broad source browsing.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain are not enough.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

If the graph is missing, explore the codebase normally — do not install or require graphify.
