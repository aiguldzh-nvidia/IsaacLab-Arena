# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live ArenaEnvGraphSpec review tool — Streamlit editor + USD thumbnails.

The CLI is a thin launcher: it always boots the Streamlit app in
``review_app.py``. The static-HTML mode and the old ``--serve`` /
``--render-thumbnails`` switches were collapsed away — thumbnails are now
always rendered (cache-hit when possible, ``SimulationApp`` cache-miss
otherwise), and the result is shown inside the live editor.

Three panels (dark dashboard style) inside the embedded view:
  * Top-left — graph diagram (mermaid.js, CDN-loaded) of the initial-state
    spatial constraints. Anchor nodes are highlighted; constraints without
    a child (is_anchor / position_limits / at_pose / ...) are listed below
    the graph rather than rendered as self-loops.
  * Bottom-left — task table (id, type, initial/success state ids, task_args).
  * Right — node card grid: type badge, asset name, and the per-node YAML
    stanza. The per-node thumbnail is a real USD viewport capture (cached
    on disk under ``.cache/llm_env_gen_thumbnails/`` and inlined as base64
    so the HTML stays self-contained).

Usage:
    /isaac-sim/python.sh -m isaaclab_arena.environments.agentic_env_gen.review_graph \\
        --yaml isaaclab_arena/tests/test_data/pick_and_place_maple_table_env_graph.yaml

    # Custom port:
    /isaac-sim/python.sh -m isaaclab_arena.environments.agentic_env_gen.review_graph \\
        --yaml <path> --port 8600

Public API used by ``review_app.py``:
    * :func:`launch_simulation_app` — boots Kit's ``SimulationApp`` (headless +
      cameras). Returns ``None`` on failure so the app can degrade to
      placeholder thumbnails rather than crashing.
    * :func:`render_thumbnails_for_spec` — given a live ``SimulationApp`` and
      a parsed spec, returns ``{node_id: png_bytes}``. Cache-aware: existing
      PNGs under the disk cache are read directly; missing ones are rendered
      through the live app and written back to the cache for next time.
    * :func:`render_html_for_spec` — full HTML payload with the given
      thumbnails dict inlined. Pass an empty dict to fall back to placeholders.

Note on USD rendering:
    ``pxr.UsdAppUtils.FrameRecorder`` and the ``usdrecord`` CLI are NOT
    available inside the Isaac Sim container (Kit ships ``UsdAppUtils.py``
    but strips out ``libusd_usdAppUtils.so``, and ``usdrecord`` is omitted
    entirely). The Kit-equivalent path used here is:
    ``omni.usd`` to open the stage + ``omni.kit.viewport.utility`` to
    capture the active viewport. Kit transparently uses cached Nucleus
    thumbnails when opening ``omniverse://`` URIs, so we don't need a
    separate Nucleus-HTTPS probe path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html as html_lib
import os
import re
import subprocess
import sys
import yaml
from dataclasses import asdict
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphSpec,
    ArenaEnvGraphStateSpec,
    _yaml_dict_factory,
)

# Disk cache for rendered thumbnails. Keyed by sha1(usd_path) so identical
# USDs across envs reuse the same PNG. Survives across runs to avoid the
# ~30s SimulationApp boot when nothing changed.
_THUMBNAIL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "llm_env_gen_thumbnails"
_THUMBNAIL_SIZE = 256


def main() -> None:
    """CLI entry point — argparse parses the user's flags, then we hand off
    to Streamlit. The actual interactive UI lives in ``review_app.py``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Path to an ArenaEnvGraphSpec YAML file. The Streamlit app will open it for live editing.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    args = parser.parse_args()
    _serve_live_editor(args.yaml, port=args.port)


def _serve_live_editor(yaml_path: Path, port: int = 8501) -> None:
    """Spawn ``streamlit run review_app.py -- --yaml <path>`` and wait.

    This is the only path through the CLI now — the old static-HTML and
    standalone ``--render-thumbnails`` flows were folded into the Streamlit
    app, which boots ``SimulationApp`` once via ``@st.cache_resource`` and
    keeps it alive for the lifetime of the server. We resolve
    ``review_app.py`` next to this file rather than going through ``-m`` so
    Streamlit picks the path up cleanly (``streamlit run`` doesn't accept
    module dotted-paths).
    """
    app_path = Path(__file__).with_name("review_app.py")
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path} — installation is incomplete.")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        # Skip the email prompt the first time Streamlit runs in a fresh
        # container — the live editor is a developer tool, not a hosted
        # service, and an interactive prompt would block automation.
        "--browser.gatherUsageStats",
        "false",
        # File watcher is a footgun here: Kit's ``SimulationApp`` boot is
        # tens of seconds; we don't want Streamlit silently rerunning the
        # script (and reissuing the cached_resource init) every time we
        # save a source file during development. The user can still hit "R"
        # in the browser to force a rerun if they want.
        "--server.fileWatcherType",
        "none",
        "--",
        "--yaml",
        str(yaml_path.resolve()),
    ]

    # Inherit env so the Streamlit subprocess sees PYTHONPATH / isaac-sim
    # site-packages exactly the same way we do.
    print(f"[review_graph] launching Streamlit live editor: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, env=os.environ.copy(), check=True)
    except FileNotFoundError as exc:
        # The plain ``pip install streamlit`` fails inside the isaaclab_arena
        # container because streamlit≥1.30 needs uvicorn>=0.30 but Kit ships
        # a bundled uvicorn==0.29 under a read-only /isaac-sim/extscache path.
        # ``--user --ignore-installed`` sidesteps the rollback by writing
        # everything to ~/.local (which is earlier on sys.path than extscache).
        raise SystemExit(
            "Streamlit is not installed. Inside the isaaclab_arena container run:\n"
            "  python -m pip install --user --ignore-installed streamlit streamlit-ace"
        ) from exc
    except KeyboardInterrupt:
        # Normal exit path — user hit Ctrl-C in the terminal.
        pass


# ---------------------------------------------------------------------------
# Public API consumed by review_app.py
# ---------------------------------------------------------------------------


def render_html_for_spec(spec: ArenaEnvGraphSpec, thumbnails: dict[str, bytes] | None = None) -> str:
    """Render the review HTML for ``spec``, inlining the given thumbnails.

    Thin public alias of :func:`_render_html` so external entry points don't
    have to reach into a private name. Pass ``thumbnails=None`` (or omit) to
    fall back to placeholder thumbnails — useful when ``SimulationApp`` is
    unavailable.
    """
    return _render_html(spec, thumbnails=thumbnails)


def launch_simulation_app():
    """Public alias for :func:`_launch_simulation_app`.

    Boots Kit's ``SimulationApp`` (headless, with cameras + UI hidden) and
    returns the live app handle. Returns ``None`` on failure so callers can
    degrade gracefully (e.g. the Streamlit app falls back to placeholder
    thumbnails instead of crashing).

    The returned app *must* be reused across thumbnail renders — Kit only
    supports one ``SimulationApp`` per process. The Streamlit app holds the
    instance under ``@st.cache_resource``.
    """
    return _launch_simulation_app()


def render_thumbnails_for_spec(app, spec: ArenaEnvGraphSpec) -> dict[str, bytes]:
    """Public alias for :func:`_render_thumbnails_with_app`.

    Resolves each node's USD via ``AssetRegistry``, reads the on-disk PNG
    cache for hits, and renders cache-misses through the live ``app``
    handle. Safe to call repeatedly across reruns — cache-hit cost is just
    a ``read_bytes()`` per node.
    """
    return _render_thumbnails_with_app(app, spec)


# ---------------------------------------------------------------------------
# Top-level HTML
# ---------------------------------------------------------------------------


def _render_html(spec: ArenaEnvGraphSpec, thumbnails: dict[str, bytes] | None = None) -> str:
    initial_state = _pick_initial_state(spec)
    thumbnails = thumbnails or {}
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
    <div class="node-grid">{_render_node_cards(spec, thumbnails)}</div>
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


def _render_node_cards(spec: ArenaEnvGraphSpec, thumbnails: dict[str, bytes]) -> str:
    return "\n".join(_render_one_node_card(node, thumbnails.get(node.id)) for node in spec.nodes)


def _render_one_node_card(node: ArenaEnvGraphNodeSpec, png_bytes: bytes | None) -> str:
    node_dict = asdict(node, dict_factory=_yaml_dict_factory)
    node_yaml = yaml.safe_dump(node_dict, sort_keys=False).rstrip()
    thumb = _render_node_thumbnail(node, png_bytes)
    return f"""<article class="node-card type-{html_lib.escape(node.type.value)}">
  {thumb}
  <div class="node-meta">
    <div class="node-id">{html_lib.escape(node.id)}</div>
    <span class="badge type-{html_lib.escape(node.type.value)}">{html_lib.escape(node.type.value)}</span>
  </div>
  <pre class="node-yaml">{html_lib.escape(node_yaml)}</pre>
</article>"""


def _render_node_thumbnail(node: ArenaEnvGraphNodeSpec, png_bytes: bytes | None = None) -> str:
    """Per-node thumbnail: real USD viewport capture if rendered, else placeholder.

    When ``png_bytes`` is provided (i.e. ``--render-thumbnails`` ran and the
    asset was successfully captured by :func:`_render_thumbnails_for_spec`),
    inline the PNG as a ``data:image/png;base64,...`` URI so the resulting
    HTML is fully self-contained — no sidecar files to keep next to the page.

    Otherwise fall back to the lightweight two-letter placeholder card, so
    a default ``python -m ... review_graph --yaml ...`` invocation still
    produces a useful page without booting Isaac Sim.
    """
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return (
            '<div class="thumb thumb-rendered">'
            f'<img src="data:image/png;base64,{b64}" alt="{html_lib.escape(node.name)} thumbnail">'
            f'<span class="thumb-name">{html_lib.escape(node.name)}</span>'
            "</div>"
        )
    initial = (node.name[:2] if node.name else "?").upper()
    return f"""<div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(node.name)}</span>
  </div>"""


# ---------------------------------------------------------------------------
# USD viewport capture (opt-in via --render-thumbnails)
# ---------------------------------------------------------------------------


def _render_thumbnails_with_app(app, spec: ArenaEnvGraphSpec) -> dict[str, bytes]:
    """Resolve each node's USD via ``AssetRegistry``, render or read cache.

    ``app`` must already be a booted ``SimulationApp``. The caller owns the
    lifecycle so the HTML write can happen before ``app.close()`` (which Kit
    may turn into ``os._exit(0)``).

    Returns ``{node.id: png_bytes}`` for nodes whose asset USD could be
    located *and* rendered. Missing entries fall through to the placeholder
    in :func:`_render_node_thumbnail`, so a partial failure (one bad asset)
    never breaks the rest of the page.

    Ordering matters: ``SimulationApp`` MUST be launched before any
    ``AssetRegistry`` access, because ``ensure_assets_registered()`` imports
    isaaclab asset modules which transitively load ``pxr``. ``pxr`` loaded
    before ``AppLauncher`` puts Kit's extension manager into an unrecoverable
    state ("extension class wrapper for base class ... has not been created
    yet"). This is the same root cause we fixed for the pytest suite.
    """
    asset_paths = _resolve_node_usd_paths(spec)
    if not asset_paths:
        print("[review_graph] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}

    _THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Split into cache-hits vs to-render. Cache key is sha1(usd_path) so
    # the same USD across multiple envs / nodes hits the same PNG.
    rendered: dict[str, bytes] = {}
    to_render: dict[str, tuple[str, Path]] = {}
    for node_id, usd_path in asset_paths.items():
        cache_path = _THUMBNAIL_CACHE_DIR / f"{_usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            rendered[node_id] = cache_path.read_bytes()
        else:
            to_render[node_id] = (usd_path, cache_path)

    if to_render:
        print(
            f"[review_graph] rendering {len(to_render)} new thumbnail(s) "
            f"(reusing {len(rendered)} from cache at {_THUMBNAIL_CACHE_DIR})...",
            file=sys.stderr,
        )
        rendered.update(_capture_usd_thumbnails(app, to_render))
    else:
        print(f"[review_graph] all {len(rendered)} thumbnail(s) served from cache.", file=sys.stderr)

    return rendered


def _launch_simulation_app():
    """Boot Isaac Sim's ``SimulationApp`` for headless viewport capture, or ``None`` on failure.

    Kept as a tiny helper so the call site can lazy-import inside this
    function — module-level import of ``simulation_app`` would drag Kit
    into every invocation, including ``--help``.
    """
    try:
        # Lazy-import: keeps the default ``review_graph`` invocation Kit-free.
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher  # noqa: PLC0415

        sim_args = argparse.Namespace(headless=True, enable_cameras=True, hide_ui=True, livestream=-1)
        return get_app_launcher(sim_args).app
    except Exception as exc:
        print(f"[review_graph] SimulationApp launch failed: {exc}", file=sys.stderr)
        return None


def _resolve_node_usd_paths(spec: ArenaEnvGraphSpec) -> dict[str, str]:
    """Map ``node.id → usd_path`` via :class:`AssetRegistry`, skipping unresolvable nodes.

    Tries two lookup strategies in order:

    1. Class-attribute ``cls.usd_path`` — the convention every ``LibraryObject``
       subclass in ``object_library.py`` follows. No instantiation, cheap.

    2. ``cls().scene_config.robot.spawn.usd_path`` — the convention every
       :class:`EmbodimentBase` subclass uses. Requires instantiating the
       embodiment because the Franka embodiments populate ``scene_config.robot``
       inside ``__init__`` rather than as a class default. Embodiment
       ``__init__`` is light (no Kit / sim required) — it only constructs
       configclass objects.

    This function MUST be called only after ``SimulationApp`` has booted — see
    the docstring of :func:`_render_thumbnails_with_app` for why.
    """
    try:
        from isaaclab_arena.assets.registries import AssetRegistry  # noqa: PLC0415
    except Exception as exc:
        print(f"[review_graph] AssetRegistry import failed: {exc}", file=sys.stderr)
        return {}

    registry = AssetRegistry()
    paths: dict[str, str] = {}
    for node in spec.nodes:
        try:
            if not registry.is_registered(node.name):
                print(f"[review_graph]   {node.id}: asset '{node.name}' not registered, skipping.", file=sys.stderr)
                continue
            cls = registry.get_asset_by_name(node.name)
            usd_path = _extract_usd_path(cls)
            if not usd_path:
                print(f"[review_graph]   {node.id}: '{node.name}' has no usd_path, skipping.", file=sys.stderr)
                continue
            paths[node.id] = usd_path
        except Exception as exc:
            print(f"[review_graph]   {node.id}: lookup failed for '{node.name}': {exc}", file=sys.stderr)
    return paths


def _extract_usd_path(cls) -> str | None:
    """Return the asset's root USD path, or ``None`` if not extractable.

    See :func:`_resolve_node_usd_paths` for the two strategies tried in order.
    """
    # Strategy 1: ``LibraryObject`` convention.
    usd_path = getattr(cls, "usd_path", None)
    if usd_path:
        return usd_path

    # Strategy 2: ``EmbodimentBase`` convention. Walk
    # ``instance.scene_config.robot.spawn.usd_path``. We instantiate with no
    # args; every embodiment ``__init__`` defaults all parameters.
    # NoEmbodiment legitimately has no robot — its instance.scene_config
    # exists but ``.robot`` is absent / None, so the getattr chain returns
    # None and we silently fall through.
    try:
        instance = cls()
    except Exception:
        return None
    scene_config = getattr(instance, "scene_config", None)
    robot = getattr(scene_config, "robot", None) if scene_config is not None else None
    spawn = getattr(robot, "spawn", None) if robot is not None else None
    return getattr(spawn, "usd_path", None) if spawn is not None else None


def _usd_cache_key(usd_path: str) -> str:
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def _capture_usd_thumbnails(app, to_render: dict[str, tuple[str, Path]]) -> dict[str, bytes]:
    """Capture all queued USDs under one already-booted ``SimulationApp``.

    Deduplicates by ``usd_path`` so the same USD shared by multiple nodes is
    only rendered once and the bytes are fanned back out.
    """
    out: dict[str, bytes] = {}

    path_to_node_ids: dict[str, list[str]] = {}
    path_to_cache: dict[str, Path] = {}
    for node_id, (usd_path, cache_path) in to_render.items():
        path_to_node_ids.setdefault(usd_path, []).append(node_id)
        path_to_cache[usd_path] = cache_path

    for usd_path, node_ids in path_to_node_ids.items():
        cache_path = path_to_cache[usd_path]
        try:
            png_bytes = _render_one_usd(app, usd_path, cache_path)
        except Exception as exc:
            print(f"[review_graph]   render failed for {usd_path}: {exc}", file=sys.stderr)
            continue
        if png_bytes:
            for node_id in node_ids:
                out[node_id] = png_bytes

    return out


def _render_one_usd(app, usd_path: str, cache_path: Path) -> bytes | None:
    """Open ``usd_path`` directly as the stage, frame the camera, capture PNG.

    Opening the USD as the stage root (rather than ``new_stage`` + reference
    wrapper) is what makes viewport capture actually produce a file in
    headless mode — Kit's viewport machinery binds to the just-opened stage
    cleanly, whereas a referenced sub-stage left the render product empty in
    every test we tried. The trade-off is that we lose isolation between
    captures (each call replaces the stage), but Kit handles that fine
    because we call ``open_stage`` again on the next asset.
    """
    import omni.usd  # noqa: PLC0415
    from omni.kit.viewport.utility import (  # noqa: PLC0415
        capture_viewport_to_file,
        frame_viewport_prims,
        get_active_viewport,
    )
    from pxr import Sdf  # noqa: PLC0415

    ctx = omni.usd.get_context()
    if not ctx.open_stage(usd_path):
        print(f"[review_graph]   open_stage failed: {usd_path}", file=sys.stderr)
        return None
    stage = ctx.get_stage()

    # Wait for textures / payloads / Nucleus fetches to settle before framing.
    _wait_for_stage_load(app, ctx)

    # Standalone object USDs (avocado, bowl, ...) ship no lights, so a viewport
    # capture renders them as a near-black silhouette against the dark skybox
    # — that's the "blank thumbnail" symptom. Complete scene USDs (maple table)
    # already include their own lighting, so this is a no-op for them.
    _ensure_default_lighting(stage)

    # Use the default prim if present, otherwise the pseudo-root, for framing.
    target_prim = stage.GetDefaultPrim()
    if not target_prim or not target_prim.IsValid():
        target_prim = stage.GetPrimAtPath(Sdf.Path("/"))

    viewport = get_active_viewport()

    # Use Kit's own ``frame_viewport_prims`` (the "F"-key equivalent / ``FramePrimsCommand``)
    # so we go through the viewport camera controller. Manually editing the
    # ``/OmniverseKit_Persp`` xform op directly worked sometimes but Kit's
    # camera controller treats /OmniverseKit_Persp as an internal state and
    # silently overrode our edits for small assets — that's why avocado / bowl
    # captured as tiny specks even with the right math. Letting Kit do the
    # framing is both correct and avoids us re-implementing the math.
    framed = frame_viewport_prims(viewport, prims=[str(target_prim.GetPath())])
    if not framed:
        print(f"[review_graph]   warning: frame_viewport_prims failed for {usd_path}", file=sys.stderr)

    # Settle Hydra after camera change so the captured frame matches the new pose.
    for _ in range(30):
        app.update()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    capture_obj = capture_viewport_to_file(viewport, str(cache_path))

    _wait_for_capture(app, capture_obj, cache_path, max_updates=600)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    print(f"[review_graph]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


def _wait_for_stage_load(app, usd_context, max_updates: int = 600) -> None:
    """Pump frames until ``usd_context.get_stage_loading_status()`` reports nothing pending.

    Returns after stage load completes or after the budget is exhausted. We
    also need a few extra frames after the count goes to zero so material
    binding / texture upload finishes — they don't show up in the load count.
    """
    settled = 0
    for _ in range(max_updates):
        app.update()
        try:
            _msg, loading_count, loaded_count = usd_context.get_stage_loading_status()
        except Exception:
            return
        if loading_count == 0 and loaded_count == 0:
            settled += 1
            if settled > 15:
                return
        else:
            settled = 0


def _wait_for_capture(app, capture_obj, cache_path: Path, max_updates: int = 600) -> None:
    """Pump ``app.update()`` until the capture PNG lands on disk (or we time out).

    Kit's capture future is fulfilled inside its async loop during
    ``app.update()``, but future completion doesn't always coincide with the
    file being flushed — checking the file directly is the most reliable
    completion signal. We also keep the future-based fast path so a
    successful capture doesn't have to wait for the file system to settle.
    """
    if capture_obj is None:
        for _ in range(max_updates):
            app.update()
        return

    future = (
        getattr(capture_obj, "_Capture__future", None)
        or getattr(capture_obj, "_RenderCapture__future", None)
        or getattr(capture_obj, "future", None)
    )

    for _ in range(max_updates):
        app.update()
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return
        if future is not None and future.done():
            # Future is done but file might still be flushing — give it a few frames.
            for _ in range(15):
                app.update()
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    return
            return


def _ensure_default_lighting(stage) -> None:
    """Add a dome + key distant light if the stage has none.

    Without this, standalone object USDs (which don't ship their own lights)
    render as a near-black silhouette. We skip the addition if any
    ``UsdLuxLight``-derived prim already exists on the stage to avoid
    double-lighting scenes like the maple table that bake in their own rig.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: PLC0415

    for prim in stage.Traverse():
        if (
            prim.HasAPI(UsdLux.LightAPI)
            or prim.IsA(UsdLux.BoundableLightBase)
            or prim.IsA(UsdLux.NonboundableLightBase)
        ):
            return

    # Soft hemispherical fill so the asset is visible from any angle, plus a
    # weak directional key for shape definition. Intensities are tuned for
    # OmniPBR / RTX defaults; tweak if asset libraries adopt darker materials.
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/_ReviewDomeLight"))
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/_ReviewKeyLight"))
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(2.0)
    # Aim the key roughly from the camera's 3/4 angle so the lit side faces
    # the viewport.
    key_xformable = UsdGeom.Xformable(key.GetPrim())
    key_xformable.ClearXformOpOrder()
    rot = key_xformable.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))


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
                    align-items: center; justify-content: center; color: var(--fg-muted);
                    position: relative; overflow: hidden; }
.node-card .thumb-rendered { background: #0e1115; }
.node-card .thumb-rendered img { width: 100%; height: 100%; object-fit: contain; display: block; }
.node-card .thumb-rendered .thumb-name { position: absolute; bottom: 0; left: 0; right: 0;
                                         padding: 4px 6px; background: rgba(15, 17, 21, 0.78);
                                         color: var(--fg); margin: 0; }
.thumb-initial { font-size: 36px; font-weight: 700; color: var(--fg); opacity: 0.6;
                 font-family: ui-monospace, monospace; }
.thumb-name { font-size: 10px; margin-top: 6px; padding: 0 8px; text-align: center; word-break: break-word; }
.node-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.node-id { font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600; word-break: break-all; }
.node-yaml { font-size: 11px; }
"""


if __name__ == "__main__":
    main()
