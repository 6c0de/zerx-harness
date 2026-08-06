"""Live + replay visualizer for zerx runs -- pure dev tooling, never
bundled into the Kaggle submission (scripts/build_notebook.py only
bundles zerx/*.py). See
docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md.

Usage:
    scripts/visualize_play.py --live --game ls20 [--max-steps 80] [--save traces/ls20.jsonl] [--history-cap 500]
    scripts/visualize_play.py --live --game ls20 --save traces/  (auto-names traces/ls20-<timestamp>.jsonl)
    scripts/visualize_play.py --replay traces/ls20-20260806T000000.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
# scripts/ has no __init__.py and this file is normally invoked directly
# (`python scripts/visualize_play.py`), where only scripts/ itself lands
# on sys.path -- ROOT must be added explicitly before any zerx/agent
# import, exactly matching scripts/play_local.py's own established
# pattern (namespace packages resolve fine once the parent dir is present,
# no __init__.py needed).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from zerx.trace import CompositeTraceRecorder, JsonlTraceWriter, TraceMeta, TraceStep  # noqa: E402

# Official ARC-AGI-3 16-color palette (indices 0-15), sourced from the
# arc_agi package's own rendering table
# (.venv/Lib/site-packages/arc_agi/rendering.py's COLOR_MAP, hex -> RGB) --
# NOT the classic 10-color ARC-AGI-1/2 puzzle palette. Real game grids use
# indices up to 15; confirmed against a real cerebras_dev trace that the
# earlier 0-9 table silently mis-colored most of a frame (index 4, the
# dominant cell value in that trace, is officially "Off Black" #333333,
# not yellow).
_PALETTE = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}
_DEFAULT_COLOR = (85, 85, 85)
_CELL_PX = 10


def _color_for(cell_value: int) -> Tuple[int, int, int]:
    return _PALETTE.get(cell_value, _DEFAULT_COLOR)


def _clamp_index(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))


def _clamp_scroll(scroll: int, total_lines: int, visible_lines: int) -> int:
    """Clamp a reasoning-panel scroll offset so the visible window never
    runs past the end of the wrapped text (or before its start)."""
    max_scroll = max(0, total_lines - visible_lines)
    return max(0, min(scroll, max_scroll))


def _wrap_reasoning(text: str, chars_per_line: int) -> List[str]:
    """Character-wrap `text` (the reasoning panel's raw model response,
    which has no guaranteed word boundaries) at `chars_per_line`. Purely a
    wrap -- the caller (`_render`) is responsible for windowing a scroll
    position over the result; this function never truncates.
    """
    chars_per_line = max(1, chars_per_line)
    if not text:
        return []
    return [text[i : i + chars_per_line] for i in range(0, len(text), chars_per_line)]


def _load_trace(path: str) -> Tuple[TraceMeta, List[TraceStep]]:
    steps: List[TraceStep] = []
    meta: Optional[TraceMeta] = None
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record_type = payload.pop("type")
            if record_type == "meta":
                meta = TraceMeta(**payload)
            elif record_type == "step":
                # JSON has no tuple type -- grid round-trips as list[list[int]];
                # restore TraceStep's declared tuple[tuple[int, ...], ...] shape.
                payload["grid"] = tuple(tuple(row) for row in payload["grid"])
                steps.append(TraceStep(**payload))
    if meta is None:
        raise ValueError(f"{path}: no meta record found")
    return meta, steps


class LivePygameRecorder:
    """Renders each recorded step to a pygame window as it arrives, keeps
    a capped in-memory history, and blocks inside `record()` while paused
    -- since `record()` runs on the real game loop's own thread (see
    agent/my_agent.py's choose_action -> Agent.main()), this genuinely
    halts play, not just the display.
    """

    def __init__(self, history_cap: int = 500) -> None:
        import pygame  # imported here, not at module scope, so pure
        # helpers above stay importable/testable without a display driver

        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption("zerx visualizer -- live")
        # 900px was too narrow for this project's real 64x64 grids: panel_x
        # lands at 64*_CELL_PX+20=660px, leaving only 240px (~26 chars at
        # this font) for reasoning text that was wrapped at a hardcoded 48
        # chars/line -- the remainder was drawn off-window and lost. 1280px
        # gives a 620px+ gutter; _render also now derives the actual wrap
        # width from real font metrics instead of a fixed guess, so this
        # stays correct if the grid or window size ever changes.
        self._screen = pygame.display.set_mode((1280, 700))
        self._font = pygame.font.SysFont("consolas", 16)
        self._history: "deque[TraceStep]" = deque(maxlen=history_cap)
        self._cursor = -1  # -1 == following live
        self._paused = False
        self._replay_mode = False  # set True by _run_replay; disables
        # SPACE's pause-toggle since replay has no running loop to pause
        self._reasoning_scroll = 0
        self._current_step: Optional[TraceStep] = None

    def record(self, step: TraceStep) -> None:
        self._history.append(step)
        self._cursor = -1
        self._render(step)
        while self._pump_events():
            pass

    def _pump_events(self) -> bool:
        """Handle one batch of pygame events; returns True if the caller
        should keep blocking (still paused), False to let the game loop
        continue.
        """
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_SPACE and not self._replay_mode:
                self._paused = not self._paused
            elif self._paused and event.key == pygame.K_LEFT:
                self._cursor = _clamp_index(
                    (self._cursor if self._cursor >= 0 else len(self._history) - 1) - 1,
                    len(self._history),
                )
                self._render(self._history[self._cursor])
            elif self._paused and event.key == pygame.K_RIGHT:
                base = self._cursor if self._cursor >= 0 else len(self._history) - 1
                self._cursor = _clamp_index(base + 1, len(self._history))
                self._render(self._history[self._cursor])
            elif self._paused and event.key == pygame.K_UP:
                self._reasoning_scroll = max(0, self._reasoning_scroll - 1)
                self._render(self._current_step, reset_scroll=False)
            elif self._paused and event.key == pygame.K_DOWN:
                self._reasoning_scroll += 1  # _render clamps via _clamp_scroll
                self._render(self._current_step, reset_scroll=False)
        self._pygame.time.wait(16)  # ~60fps ceiling; avoids a CPU-pinning
        # busy-spin in both this pause loop and _run_replay's event loop
        return self._paused

    def _render(self, step: TraceStep, *, reset_scroll: bool = True) -> None:
        screen, font = self._screen, self._font
        if reset_scroll:
            self._reasoning_scroll = 0
        self._current_step = step
        screen.fill((20, 20, 20))
        for y, row in enumerate(step.grid):
            for x, value in enumerate(row):
                rect = (x * _CELL_PX, y * _CELL_PX, _CELL_PX, _CELL_PX)
                self._pygame.draw.rect(screen, _color_for(value), rect)
        panel_x = len(step.grid[0]) * _CELL_PX + 20 if step.grid else 20
        header = [
            f"step {step.step_index}  game {step.game_id}",
            f"action {step.action_name} ({step.action_x}, {step.action_y})",
            f"source {step.source}  repaired {step.repaired}",
            f"state {step.game_state}  levels {step.levels_completed}",
            "",
            "reasoning: (up/down to scroll)",
        ]
        top_margin, right_margin, line_pitch = 10, 10, 20
        screen_width, screen_height = screen.get_size()
        # Real font metrics, not another hardcoded guess -- this is a
        # monospace font (consolas) so every glyph is the same width.
        char_px = max(font.size("X")[0], 1)
        available_px = max(screen_width - panel_x - right_margin, 0)
        chars_per_line = max(1, available_px // char_px)
        max_total_lines = max(1, (screen_height - top_margin) // line_pitch)
        max_reasoning_lines = max(0, max_total_lines - len(header))

        all_lines = _wrap_reasoning(step.reasoning, chars_per_line)
        self._reasoning_scroll = _clamp_scroll(
            self._reasoning_scroll, len(all_lines), max_reasoning_lines
        )
        visible = all_lines[self._reasoning_scroll : self._reasoning_scroll + max_reasoning_lines]
        if len(all_lines) > max_reasoning_lines:
            shown_end = min(self._reasoning_scroll + max_reasoning_lines, len(all_lines))
            header[-1] = (
                f"reasoning: (up/down to scroll) "
                f"[{self._reasoning_scroll + 1}-{shown_end}/{len(all_lines)}]"
            )

        lines = header + visible
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (230, 230, 230)), (panel_x, top_margin + i * line_pitch))
        self._pygame.display.flip()


def _run_live(args: argparse.Namespace) -> None:
    import arc_agi
    from arc_agi import OperationMode

    from agent.my_agent import MyAgent as MyAgentCls

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    if env is None:
        raise SystemExit(f"arcade.make({args.game!r}) returned None -- game unavailable")

    agent = MyAgentCls(
        card_id="visualize-play",
        game_id=args.game,
        agent_name=f"visualize-play.{args.game}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
    )
    if args.max_steps:
        agent.MAX_ACTIONS = min(agent.MAX_ACTIONS, args.max_steps)

    live_recorder = LivePygameRecorder(history_cap=args.history_cap)
    if args.save:
        writer = JsonlTraceWriter(args.save)
        writer.write_meta(
            TraceMeta(
                game_id=args.game,
                seed=0,
                backend=agent._config.backend,
                config_hash=agent._config.config_hash(),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        agent.trace_recorder = CompositeTraceRecorder([live_recorder, writer])
    else:
        agent.trace_recorder = live_recorder

    agent.main()

    # Without this, the window disappears the instant the run ends (process
    # exit), giving no chance to inspect the final frame/decision unless
    # SPACE happened to be pressed before the last step landed. Force-pause
    # and keep pumping events -- reuses the existing pause/history-nav/QUIT
    # path, so LEFT/RIGHT history navigation and closing the window both
    # keep working exactly as they do mid-run.
    live_recorder._paused = True
    while live_recorder._pump_events():
        pass


def _run_replay(args: argparse.Namespace) -> None:
    meta, steps = _load_trace(args.replay)
    recorder = LivePygameRecorder(history_cap=max(len(steps), 1))
    recorder._paused = True  # replay is always "paused": step-only, no running loop
    recorder._replay_mode = True  # SPACE is a no-op; there's no live loop to pause
    for step in steps:
        recorder._history.append(step)
    if steps:
        recorder._cursor = 0
        recorder._render(steps[0])
        while True:
            recorder._pump_events()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--game")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save")
    parser.add_argument("--history-cap", type=int, default=500)
    parser.add_argument("--replay")
    args = parser.parse_args()

    if args.live:
        if not args.game:
            parser.error("--live requires --game")
        _run_live(args)
    elif args.replay:
        _run_replay(args)
    else:
        parser.error("pass --live --game <id> or --replay <trace.jsonl>")


if __name__ == "__main__":
    main()
