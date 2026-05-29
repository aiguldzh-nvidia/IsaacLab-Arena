# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Streamlit-backed live editor for ArenaEnvGraphSpec YAMLs.

Wraps :func:`isaaclab_arena.environments.agentic_env_gen.review_graph.render_html_for_spec`
in a two-pane Streamlit page so the user can edit the YAML in the browser and
re-render the dark-dashboard visualization on demand. ``SimulationApp`` is
booted once via ``@st.cache_resource`` and reused for every thumbnail render
(disk-cache hit when possible, live USD viewport capture when not).

Launch (always via the wrapper in review_graph.py — handles streamlit flags):
    /isaac-sim/python.sh -m isaaclab_arena.environments.agentic_env_gen.review_graph \\
        --yaml path/to/spec.yaml

Design:
  * Left pane — ``streamlit-ace`` YAML editor + validation badge + action
    buttons. Validation runs on every rerun (i.e. after each editor blur).
  * Right pane — sandboxed iframe with the rendered review HTML. Updates only
    when the user clicks "Regenerate", so editing never causes mid-typing
    re-renders.
  * Thumbnails — real USD viewport captures. Booted ``SimulationApp`` lives
    inside an ``@st.cache_resource`` so its ~30s startup is paid once per
    server lifetime. PNGs are cached on disk under
    ``.cache/llm_env_gen_thumbnails/`` and survive across runs.
"""

from __future__ import annotations

import argparse
import traceback
import yaml
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from isaaclab_arena.environments.agentic_env_gen.review_graph import (
    launch_simulation_app,
    render_html_for_spec,
    render_thumbnails_for_spec,
)
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec

# Visualization iframe height. Tuned so the graph + tasks + node grid all
# fit without an outer Streamlit scrollbar swallowing the inner one.
_IFRAME_HEIGHT_PX = 1100


# ---------------------------------------------------------------------------
# SimulationApp lifecycle
# ---------------------------------------------------------------------------


def _patch_signal_for_non_main_thread() -> None:
    """No-op ``signal.signal`` for non-main threads so Kit's bootstrap doesn't raise.

    Kit's ``SimulationApp.__init__`` installs SIGINT/SIGTERM handlers via
    ``signal.signal``, which raises ``ValueError("signal only works in main
    thread of the main interpreter")`` when called outside the interpreter's
    main thread. Streamlit's ``ScriptRunner`` runs the user script in a
    worker thread per session — so the SimApp boot would crash there even
    though it would succeed under a plain ``python -m`` invocation.

    We swallow *only* the "main thread" ValueError and let everything else
    propagate. The Streamlit main thread keeps its own SIGINT handler (it's
    installed before our script ever runs), so Ctrl-C still tears the server
    down cleanly. The patch is idempotent — repeated calls re-wrap the
    already-wrapped function harmlessly.
    """
    import signal  # noqa: PLC0415

    if getattr(signal.signal, "_arena_patched", False):
        return

    _original_signal = signal.signal

    def _safe_signal(signum, handler):
        try:
            return _original_signal(signum, handler)
        except ValueError as exc:
            if "main thread" in str(exc):
                return None
            raise

    _safe_signal._arena_patched = True  # type: ignore[attr-defined]
    signal.signal = _safe_signal  # type: ignore[assignment]


@st.cache_resource(show_spinner="Booting Isaac Sim (≈30s on first run, cached afterwards)…")
def _get_simulation_app():
    """Boot ``SimulationApp`` exactly once per Streamlit server process.

    ``@st.cache_resource`` is exactly the right primitive here: the returned
    handle is shared across reruns and across browser sessions, and Streamlit
    won't try to pickle it (unlike ``@st.cache_data``). If the boot fails we
    return ``None`` and the app degrades to placeholder thumbnails — the
    review page still works, just with the styled two-letter cards instead
    of real USD captures.

    The signal patch must happen *before* the SimApp launch (which happens
    on the Streamlit worker thread); see ``_patch_signal_for_non_main_thread``.
    """
    _patch_signal_for_non_main_thread()
    return launch_simulation_app()


def _render_with_thumbnails(spec: ArenaEnvGraphSpec) -> str:
    """Render the review HTML for ``spec`` with thumbnails resolved through
    the cached ``SimulationApp``.

    Cache-aware in two layers:
      * The disk cache under ``.cache/llm_env_gen_thumbnails/`` survives
        across runs; ``render_thumbnails_for_spec`` reads it directly.
      * Within a run, ``@st.cache_resource`` ensures we never re-boot Kit.

    If ``SimulationApp`` is unavailable (boot failed) we fall back to
    placeholder thumbnails so the user still gets a usable page.
    """
    app = _get_simulation_app()
    if app is None:
        st.warning(
            "Isaac Sim is unavailable — falling back to placeholder thumbnails. "
            "Check the terminal where you launched the server for the boot error.",
            icon="⚠️",
        )
        return render_html_for_spec(spec, thumbnails=None)
    thumbnails = render_thumbnails_for_spec(app, spec)
    return render_html_for_spec(spec, thumbnails=thumbnails)


# ---------------------------------------------------------------------------
# Args + session-state init
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Pull ``--yaml`` from ``sys.argv`` (post ``--`` from the streamlit CLI).

    Streamlit forwards anything after ``--`` on its command line into the
    script's ``sys.argv``. We use a tolerant parser so reruns (which keep
    argv intact) never abort the app.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yaml", type=Path, required=True)
    args, _unknown = parser.parse_known_args()
    return args


@dataclass
class _ValidationResult:
    """Outcome of validating editor text against ``ArenaEnvGraphSpec``."""

    spec: ArenaEnvGraphSpec | None
    error: str | None  # human-readable, multi-line; None iff spec is not None

    @property
    def is_valid(self) -> bool:
        return self.spec is not None


def _validate_yaml_text(text: str) -> _ValidationResult:
    """Two-stage validation: yaml.safe_load → ArenaEnvGraphSpec.from_dict.

    Returns a populated error string on the first failing stage so the UI can
    render exactly one red banner with the most actionable message.
    """
    # Stage 1: YAML parse. PyYAML's ``problem_mark`` is the most useful
    # location info we get for syntax errors — surface it explicitly.
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        loc = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
        return _ValidationResult(spec=None, error=f"YAML parse error{loc}:\n{exc}")

    if not isinstance(loaded, dict):
        return _ValidationResult(
            spec=None,
            error=f"Top-level YAML must be a mapping, got {type(loaded).__name__}.",
        )

    # Stage 2: schema validation via the existing dataclass parser.
    try:
        spec = ArenaEnvGraphSpec.from_dict(loaded)
    except Exception as exc:
        # ``from_dict`` raises a mix of AssertionError / ValueError / KeyError
        # depending on which graph_spec_utils check trips. Keep the type so
        # users can tell schema errors apart from YAML errors.
        tb = traceback.format_exception_only(type(exc), exc)
        return _ValidationResult(spec=None, error="".join(tb).rstrip())

    return _ValidationResult(spec=spec, error=None)


def _initialize_state(yaml_path: Path) -> None:
    """Seed ``st.session_state`` from disk exactly once per session.

    We key off ``_yaml_path`` so that if the user passes a different YAML on
    a Streamlit reload (rare — usually the same), we reset cleanly.
    """
    if st.session_state.get("_yaml_path") == str(yaml_path):
        return

    original_text = yaml_path.read_text(encoding="utf-8")

    st.session_state["_yaml_path"] = str(yaml_path)
    st.session_state["original_text"] = original_text
    st.session_state["edited_text"] = original_text
    # The text whose render is currently displayed. Starts == original so the
    # first paint shows the on-disk file (and "Regenerate" is correctly
    # disabled until the user edits something).
    st.session_state["last_rendered_text"] = original_text
    st.session_state["save_path"] = str(yaml_path)

    initial = _validate_yaml_text(original_text)
    if not initial.is_valid:
        # Defensive: if the on-disk file is already broken we still want to
        # show *something*, but we won't pre-render it. The user fixes the
        # YAML in the editor, then hits Regenerate.
        st.session_state["rendered_html"] = _BROKEN_PLACEHOLDER_HTML
    else:
        # First render boots ``SimulationApp`` (≈30s) via the cached resource
        # and renders any uncached USD thumbnails. Both steps are amortized
        # across subsequent regenerations.
        st.session_state["rendered_html"] = _render_with_thumbnails(initial.spec)


# Tiny standalone HTML used when the on-disk YAML is itself invalid.
_BROKEN_PLACEHOLDER_HTML = """<!DOCTYPE html><html><body style="
    font-family: ui-monospace, monospace;
    background:#15181d; color:#e4e6eb; padding:24px; margin:0;">
<p>No visualization yet — fix YAML and click <em>Regenerate</em>.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# UI panels
# ---------------------------------------------------------------------------


def _render_validation_badge(validation: _ValidationResult) -> None:
    """Show a green tick + summary, or a red cross + the raw exception text."""
    if validation.is_valid:
        spec = validation.spec
        st.success(
            f"Valid spec — {spec.env_name} · {len(spec.nodes)} nodes · "
            f"{len(spec.tasks)} tasks · {len(spec.state_specs)} state specs",
            icon="✅",
        )
    else:
        st.error(f"Invalid YAML\n\n```\n{validation.error}\n```", icon="🛑")


def _render_action_buttons(validation: _ValidationResult) -> None:
    """Render the Regenerate + Save buttons with the gating rules from the spec.

    Gating (per the PR brief):
      * Regenerate — only when edited since last render AND valid.
      * Save — only when YAML is valid.
    """
    edited_since_render = st.session_state["edited_text"] != st.session_state["last_rendered_text"]
    can_regenerate = edited_since_render and validation.is_valid
    can_save = validation.is_valid

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button(
            "Regenerate visualization",
            type="primary",
            disabled=not can_regenerate,
            use_container_width=True,
            help=(
                "Re-renders the right pane from the current editor text. "
                "Enabled only when the YAML has been edited since the last "
                "render and currently validates."
            ),
        ):
            assert validation.spec is not None  # guarded by can_regenerate
            # ``_render_with_thumbnails`` reuses the cached ``SimulationApp``
            # and the on-disk PNG cache — so a regen that doesn't introduce
            # new asset names is near-instant; one that does pays roughly
            # ~2s per new USD.
            with st.spinner("Rendering thumbnails…"):
                st.session_state["rendered_html"] = _render_with_thumbnails(validation.spec)
            st.session_state["last_rendered_text"] = st.session_state["edited_text"]
            st.toast("Visualization regenerated.", icon="🔄")
            st.rerun()

    with col_b:
        save_path_str = st.session_state["save_path"]
        if st.button(
            f"Save to {Path(save_path_str).name}",
            disabled=not can_save,
            use_container_width=True,
            help=f"Writes the editor contents to {save_path_str}. Disabled while YAML is invalid.",
        ):
            try:
                Path(save_path_str).write_text(st.session_state["edited_text"], encoding="utf-8")
                # Update "original" so future "edited since…" comparisons are
                # against the just-saved file, not the very first load.
                st.session_state["original_text"] = st.session_state["edited_text"]
                st.toast(f"Saved → {save_path_str}", icon="💾")
            except OSError as exc:
                st.error(f"Save failed: {exc}", icon="🛑")

    with st.expander("Change save location", expanded=False):
        new_path = st.text_input(
            "Save path",
            value=save_path_str,
            key="save_path_input",
            help="Defaults to the YAML file passed via --yaml.",
        )
        if new_path and new_path != save_path_str:
            st.session_state["save_path"] = new_path


def _render_editor_panel(yaml_path: Path) -> _ValidationResult:
    """Left pane. Returns the validation result for the current editor text.

    Returning the validation result (rather than stashing it in session_state)
    keeps the data flow inside one render pass and avoids a stale-state class
    of bug where the badge and the buttons disagree.
    """
    # Lazy import so the module is importable from environments that don't
    # have streamlit-ace installed yet (we surface a clean error message
    # rather than ImportError at module load).
    try:
        from streamlit_ace import st_ace  # noqa: PLC0415
    except ImportError as exc:
        # See review_graph._serve_live_editor for why --user --ignore-installed
        # is required inside the isaaclab_arena container.
        st.error(
            "`streamlit-ace` is not installed. Inside the isaaclab_arena container run:\n"
            "`python -m pip install --user --ignore-installed streamlit-ace`\n\n"
            f"Underlying error: {exc}",
            icon="🛑",
        )
        st.stop()

    st.subheader("YAML editor")
    st.caption(f"Source: `{yaml_path}`")

    # ``auto_update=False`` (the default) commits on blur / Ctrl+Enter rather
    # than on every keystroke. That gives us "live enough" validation without
    # hammering Streamlit's rerun loop while the user is mid-line.
    new_text = st_ace(
        value=st.session_state["edited_text"],
        language="yaml",
        theme="monokai",
        keybinding="vscode",
        font_size=13,
        tab_size=2,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=False,
        min_lines=30,
        # Key bound to the YAML path so swapping --yaml between sessions
        # forces ace to remount with the new content.
        key=f"ace_editor::{yaml_path}",
    )
    if new_text is not None:
        st.session_state["edited_text"] = new_text

    validation = _validate_yaml_text(st.session_state["edited_text"])
    _render_validation_badge(validation)
    _render_action_buttons(validation)
    return validation


def _render_visualization_panel() -> None:
    """Right pane — iframe-mount the cached rendered HTML."""
    st.subheader("Visualization")
    edited_since_render = st.session_state["edited_text"] != st.session_state["last_rendered_text"]
    if edited_since_render:
        st.caption("⚠️ Editor has unrendered changes — click **Regenerate visualization** to refresh.")
    else:
        st.caption("Showing the current YAML.")

    # ``st.components.v1.html`` wraps the payload in a sandboxed iframe, which
    # is what we want — the mermaid CDN script and the static CSS stay
    # isolated from Streamlit's own DOM.
    st.components.v1.html(
        st.session_state["rendered_html"],
        height=_IFRAME_HEIGHT_PX,
        scrolling=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ArenaEnvGraphSpec live editor",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    args = _parse_args()
    yaml_path = args.yaml.resolve()
    if not yaml_path.exists():
        st.error(f"YAML file not found: {yaml_path}", icon="🛑")
        st.stop()

    _initialize_state(yaml_path)

    st.markdown("### ArenaEnvGraphSpec live editor")
    left, right = st.columns([2, 3], gap="large")
    with left:
        _render_editor_panel(yaml_path)
    with right:
        _render_visualization_panel()


# Streamlit invokes the script top-level on every rerun, so we run main()
# unconditionally. The standard ``if __name__ == "__main__"`` guard would
# also work under ``streamlit run`` but is unnecessary — this module is only
# ever loaded as the Streamlit entrypoint.
main()
