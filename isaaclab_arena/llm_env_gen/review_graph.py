# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render an ArenaEnvGraphSpec YAML to a self-contained HTML review page.

Three panels (dark dashboard style):
  * Top-left — graph diagram (mermaid.js, CDN-loaded) of the initial-state
    spatial constraints. Anchor nodes are highlighted; constraints without
    a child (is_anchor / position_limits / at_pose / ...) are listed below
    the graph rather than rendered as self-loops.
  * Bottom-left — task table (id, type, initial/success state ids, task_args).
  * Right — node card grid: type badge, asset name, and the per-node YAML
    stanza. ``_render_node_thumbnail`` is the single integration point for
    a future real USD-snapshot renderer (e.g. ``pxr.UsdAppUtils.FrameRecorder``
    / ``usdrecord``); v1 emits a styled placeholder so the script stays
    lightweight and runs outside the Isaac Sim Docker container.

Usage:
    # Default: writes <yaml_stem>.html alongside the input file.
    python -m isaaclab_arena.llm_env_gen.review_graph \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml

    # Explicit output path:
    python -m isaaclab_arena.llm_env_gen.review_graph \\
        --yaml isaaclab_arena_environments/llm_generated/<env>_proposal.yaml \\
        --out /tmp/review.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import webbrowser
import yaml
from dataclasses import asdict
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    _yaml_dict_factory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", type=Path, required=True, help="Path to an ArenaEnvGraphSpec YAML file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <yaml_stem>.html next to the input.",
    )
    parser.add_argument("--open", action="store_true", help="Open the resulting HTML in the default browser.")
    args = parser.parse_args()

    spec = ArenaEnvGraphSpec.from_yaml(args.yaml)
    out_path = args.out or args.yaml.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(spec), encoding="utf-8")
    print(f"Wrote {out_path}")
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


# ---------------------------------------------------------------------------
# Top-level HTML
# ---------------------------------------------------------------------------


def _render_html(spec: ArenaEnvGraphSpec) -> str:
    initial_state = _pick_initial_state(spec)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(spec.env_name)} — graph review</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{html_lib.escape(spec.env_name)}</h1>
  <p class="sub">{len(spec.nodes)} nodes · {len(spec.tasks)} tasks · {len(spec.state_specs)} state specs</p>
</header>
<main>
  <section class="panel graph-panel">
    <h2>Spatial graph <span class="muted">(initial state: <code>{
        html_lib.escape(initial_state.id if initial_state else "<none>")
    }</code>)</span></h2>
    <pre class="mermaid">{_render_mermaid(spec, initial_state)}</pre>
    {_render_unary_constraints(initial_state)}
  </section>
  <section class="panel tasks-panel">
    <h2>Tasks</h2>
    {_render_tasks_table(spec)}
  </section>
  <section class="panel nodes-panel">
    <h2>Nodes</h2>
    <div class="node-grid">{_render_node_cards(spec)}</div>
  </section>
</main>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'dark', themeVariables: {{ fontFamily: 'ui-monospace, monospace' }} }});</script>
</body>
</html>
"""


def _pick_initial_state(spec: ArenaEnvGraphSpec) -> ArenaEnvGraphStateSpec | None:
    """Pick the state spec that tasks point at as their initial state.

    Falls back to the first state spec in the list. Returns ``None`` only if
    there are no state specs at all.
    """
    if spec.tasks:
        target_id = spec.tasks[0].initial_state_spec_id
        for s in spec.state_specs:
            if s.id == target_id:
                return s
    return spec.state_specs[0] if spec.state_specs else None


# ---------------------------------------------------------------------------
# Mermaid graph rendering
# ---------------------------------------------------------------------------


def _render_mermaid(spec: ArenaEnvGraphSpec, state: ArenaEnvGraphStateSpec | None) -> str:
    """Emit a left-to-right mermaid graph of the spatial constraints with children.

    Constraints without a child (is_anchor / position_limits / at_pose / ...)
    are dropped here and surfaced separately by :func:`_render_unary_constraints`.
    """
    lines = ["graph LR"]
    if state is None:
        lines.append("  empty[no state spec]")
        return "\n".join(lines)

    anchor_ids: set[str] = set()
    edge_nodes: set[str] = set()

    for c in state.spatial_constraints:
        kind = c.type.value
        if kind == "is_anchor":
            anchor_ids.add(c.parent)
        elif c.child is not None:
            lines.append(
                f"  {_mermaid_id(c.child)}[{_mermaid_label(c.child)}]"
                f" -->|{kind}| "
                f"{_mermaid_id(c.parent)}[{_mermaid_label(c.parent)}]"
            )
            edge_nodes.add(c.child)
            edge_nodes.add(c.parent)

    # Include every node from the spec so disconnected ones still appear.
    for node in spec.nodes:
        if node.id not in edge_nodes:
            lines.append(f"  {_mermaid_id(node.id)}[{_mermaid_label(node.id)}]")

    # Anchor highlight.
    for anchor_id in anchor_ids:
        lines.append(f"  style {_mermaid_id(anchor_id)} fill:#3a7d44,color:#fff,stroke:#7fd17f,stroke-width:2px")

    # Color nodes by type for quick visual scanning.
    type_palette = {
        "background": ("#3a4f7a", "#7aa0d8"),
        "embodiment": ("#7a3a3a", "#d87a7a"),
        "object": ("#7a6b3a", "#d8c47a"),
        "object_reference": ("#6b3a7a", "#c47ad8"),
        "lighting": ("#3a7a7a", "#7ad8d8"),
    }
    for node in spec.nodes:
        if node.id in anchor_ids:
            continue  # anchor style wins
        fill, stroke = type_palette.get(node.type.value, ("#3a3d44", "#888"))
        lines.append(f"  style {_mermaid_id(node.id)} fill:{fill},color:#fff,stroke:{stroke}")

    return "\n".join(lines)


_MERMAID_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _mermaid_id(s: str) -> str:
    """Mermaid node identifiers must be alphanumeric / underscore."""
    return _MERMAID_ID_SAFE.sub("_", s)


def _mermaid_label(s: str) -> str:
    """Escape mermaid-significant characters inside node labels."""
    return s.replace('"', "&quot;").replace("|", "&#124;")


def _render_unary_constraints(state: ArenaEnvGraphStateSpec | None) -> str:
    """List constraints without a child below the graph (anchors, position_limits, ...)."""
    if state is None:
        return ""
    rows = []
    for c in state.spatial_constraints:
        if c.child is not None:
            continue
        params = (
            f' <code class="muted">{html_lib.escape(yaml.safe_dump(c.params, default_flow_style=True).rstrip())}</code>'
            if c.params
            else ""
        )
        rows.append(
            f'<li><span class="badge type-{html_lib.escape(c.type.value)}">{html_lib.escape(c.type.value)}</span>'
            f" on <code>{html_lib.escape(c.parent)}</code>{params}</li>"
        )
    if not rows:
        return ""
    return (
        f'<details open class="unary"><summary>Unary constraints ({len(rows)})</summary>'
        f'<ul>{"".join(rows)}</ul></details>'
    )


# ---------------------------------------------------------------------------
# Tasks panel
# ---------------------------------------------------------------------------


def _render_tasks_table(spec: ArenaEnvGraphSpec) -> str:
    if not spec.tasks:
        return "<p class='muted'><em>No tasks defined.</em></p>"
    rows = []
    for t in spec.tasks:
        task_args_str = yaml.safe_dump(t.task_args, sort_keys=False).rstrip() if t.task_args else "(empty)"
        rows.append(
            "<tr>"
            f"<td><code>{html_lib.escape(t.id)}</code></td>"
            f'<td><span class="badge type-task">{html_lib.escape(t.type)}</span></td>'
            f"<td><code>{html_lib.escape(t.initial_state_spec_id)}</code></td>"
            f"<td><code>{html_lib.escape(t.success_state_spec_id) or '<em>unset</em>'}</code></td>"
            f"<td><pre>{html_lib.escape(task_args_str)}</pre></td>"
            "</tr>"
        )
    return (
        "<table class='tasks'>"
        "<thead><tr><th>id</th><th>type</th><th>initial</th><th>success</th><th>task_args</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Node cards
# ---------------------------------------------------------------------------


def _render_node_cards(spec: ArenaEnvGraphSpec) -> str:
    return "\n".join(_render_one_node_card(node) for node in spec.nodes)


def _render_one_node_card(node: ArenaEnvGraphNodeSpec) -> str:
    node_dict = asdict(node, dict_factory=_yaml_dict_factory)
    node_yaml = yaml.safe_dump(node_dict, sort_keys=False).rstrip()
    thumb = _render_node_thumbnail(node)
    return f"""<article class="node-card type-{html_lib.escape(node.type.value)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(node.id)}</div>
    <span class="badge type-{html_lib.escape(node.type.value)}">{html_lib.escape(node.type.value)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""


def _render_node_thumbnail(node: ArenaEnvGraphNodeSpec) -> str:
    """Single integration point for per-node preview rendering.

    Currently emits a styled placeholder. To wire in real USD snapshots, look
    up the asset's USD path via ``AssetRegistry.get_asset_by_name(node.name)``,
    render a PNG with ``pxr.UsdAppUtils.FrameRecorder`` (or shell out to the
    ``usdrecord`` CLI), and return an ``<img src="data:image/png;base64,...">``
    instead. The rest of the layout doesn't need to change.
    """
    initial = (node.name[:2] if node.name else "?").upper()
    return f"""<div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
  </div>"""


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #15181d;
  --bg-elev: #1d2128;
  --bg-elev2: #262b34;
  --border: #2f343d;
  --fg: #e4e6eb;
  --fg-muted: #8a9099;
  --accent: #7fd17f;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; font: 14px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--fg); }
header { margin-bottom: 16px; }
header h1 { margin: 0; font-size: 28px; font-weight: 700; }
header .sub { margin: 4px 0 0; color: var(--fg-muted); font-size: 13px; }
main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto;
       grid-template-areas: "graph nodes" "tasks nodes"; gap: 16px; }
.graph-panel { grid-area: graph; }
.tasks-panel { grid-area: tasks; }
.nodes-panel { grid-area: nodes; }
.panel { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.panel h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
.panel h2 .muted { color: var(--fg-muted); font-weight: 400; font-size: 13px; }
code { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
       background: var(--bg-elev2); padding: 1px 6px; border-radius: 4px; }
pre { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
      background: var(--bg-elev2); padding: 10px 12px; border-radius: 6px; margin: 0;
      white-space: pre-wrap; word-break: break-word; }
.muted { color: var(--fg-muted); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
         font-weight: 600; letter-spacing: 0.03em; background: var(--bg-elev2); color: var(--fg); }
.badge.type-background { background: #3a4f7a; }
.badge.type-embodiment { background: #7a3a3a; }
.badge.type-object { background: #7a6b3a; }
.badge.type-object_reference { background: #6b3a7a; }
.badge.type-lighting { background: #3a7a7a; }
.badge.type-is_anchor { background: #3a7d44; }
.badge.type-position_limits, .badge.type-at_pose, .badge.type-at_position { background: #6b3a7a; }
.badge.type-task { background: #2f343d; border: 1px solid #4a5; color: var(--accent); }
.mermaid { background: var(--bg-elev2); padding: 8px; border-radius: 6px; min-height: 220px;
           display: flex; align-items: center; justify-content: center; }
.unary { margin-top: 12px; }
.unary summary { cursor: pointer; color: var(--fg-muted); font-size: 13px; padding: 4px 0; }
.unary ul { margin: 8px 0 0; padding-left: 20px; list-style: disc; color: var(--fg); }
.unary li { padding: 3px 0; }
table.tasks { width: 100%; border-collapse: collapse; }
table.tasks th, table.tasks td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
                                  vertical-align: top; font-size: 12px; }
table.tasks th { color: var(--fg-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
table.tasks pre { padding: 6px 8px; font-size: 11px; }
.node-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.node-card { background: var(--bg-elev2); border: 1px solid var(--border); border-radius: 8px;
             padding: 12px; display: flex; flex-direction: column; gap: 10px; }
.node-card .thumb { aspect-ratio: 1 / 1; background: linear-gradient(135deg, #2a2f37, #1c2026);
                    border-radius: 6px; display: flex; flex-direction: column;
                    align-items: center; justify-content: center; color: var(--fg-muted); position: relative; }
.thumb-initial { font-size: 36px; font-weight: 700; color: var(--fg); opacity: 0.6;
                 font-family: ui-monospace, monospace; }
.thumb-name { font-size: 10px; margin-top: 6px; padding: 0 8px; text-align: center; word-break: break-word; }
.node-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.node-id { font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600; word-break: break-all; }
.node-yaml { font-size: 11px; }
"""


if __name__ == "__main__":
    main()
