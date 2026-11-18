"""Rich-formatted ArgumentParser for `send` and `runs` CLIs.

Provides grouped rounded panels, clean usage banners, syntax-highlighted examples,
and styled error output using only `rich` (zero extra dependencies) while keeping
100% standard `argparse` parsing and exit-code semantics.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Default argparse group titles that get normalized to cleaner labels
_DEFAULT_POS_TITLES = {"positional arguments"}
_DEFAULT_OPT_TITLES = {"options", "optional arguments"}


def _format_metavar(action: argparse.Action) -> Tuple[str, Optional[str]]:
    """Return `(short_metavar, choices_note)` for an argparse option/positional."""
    if action.nargs == 0 or isinstance(
        action,
        (
            argparse._HelpAction,
            argparse._StoreTrueAction,
            argparse._StoreFalseAction,
            argparse._CountAction,
            argparse._VersionAction,
        ),
    ):
        return "", None

    if action.choices is not None:
        choices_list = [str(c) for c in action.choices]
        joined = "|".join(choices_list)
        if len(joined) <= 18:
            return f"<{joined}>", None
        mv = action.metavar if isinstance(action.metavar, str) else "choice"
        return f"<{mv}>", ", ".join(choices_list)

    if isinstance(action.metavar, str) and action.metavar:
        mv = action.metavar
        if action.nargs in ("+", "*"):
            return f"<{mv}...>", None
        return f"<{mv}>", None

    if action.type is int:
        base = "int"
    elif action.type is float:
        base = "float"
    else:
        base = "str"

    if action.nargs in ("+", "*"):
        return f"<{base}...>", None
    if action.nargs == "?":
        return f"[{base}]", None
    return f"<{base}>", None


def _format_help_text(action: argparse.Action, choices_note: Optional[str]) -> Text:
    """Build styled help text with dim metadata badges (choices, default, repeatable)."""
    help_str = (action.help or "").strip()
    txt = Text(help_str)

    badges: List[str] = []
    if choices_note:
        badges.append(f"choices: {choices_note}")

    is_bool = isinstance(
        action,
        (
            argparse._HelpAction,
            argparse._StoreTrueAction,
            argparse._StoreFalseAction,
            argparse._CountAction,
        ),
    )
    if (
        not is_bool
        and action.default is not None
        and action.default != ""
        and action.default != []
        and action.default != ()
        and action.default is not False
        and action.default is not argparse.SUPPRESS
        and "default:" not in help_str.lower()
        and "(default)" not in help_str.lower()
    ):
        badges.append(f"default: {action.default}")

    if isinstance(action, argparse._AppendAction) and "repeatable" not in help_str.lower():
        badges.append("repeatable")

    if action.required and action.option_strings:
        badges.append("required")

    if badges:
        if help_str:
            txt.append("  ")
        txt.append("[" + " | ".join(badges) + "]", style="dim")

    return txt


def _parse_subcommand_entry(choice_action: argparse.Action) -> Tuple[str, List[str], str]:
    """Extract `(name, aliases, help)` from an argparse subparser choice action."""
    name = choice_action.dest
    metavar = choice_action.metavar or name
    aliases: List[str] = []
    if "(" in metavar and metavar.endswith(")"):
        inner = metavar[metavar.index("(") + 1 : -1]
        aliases = [a.strip() for a in inner.split(",") if a.strip()]
    help_text = (choice_action.help or "").strip()
    return name, aliases, help_text


def _build_usage_text(parser: "RichArgumentParser") -> Text:
    """Build a concise, highlighted `Usage:` line."""
    txt = Text()
    txt.append("Usage: ", style="bold yellow")

    if parser.usage:
        txt.append(parser.usage, style="bold white")
        return txt

    txt.append(parser.prog, style="bold cyan")

    has_subparsers = any(isinstance(a, argparse._SubParsersAction) for a in parser._actions)
    positionals = [
        a
        for a in parser._actions
        if not a.option_strings
        and not isinstance(a, argparse._SubParsersAction)
        and a.help is not argparse.SUPPRESS
    ]
    has_options = any(a.option_strings and a.help is not argparse.SUPPRESS for a in parser._actions)

    if has_subparsers:
        txt.append(" [COMMAND]", style="bold magenta")
    for pos in positionals:
        label = pos.metavar if isinstance(pos.metavar, str) and pos.metavar else pos.dest
        if pos.nargs == "?":
            txt.append(f" [{label}]", style="yellow")
        elif pos.nargs == "*":
            txt.append(f" [{label}...]", style="yellow")
        elif pos.nargs == "+":
            txt.append(f" <{label}...>", style="yellow")
        else:
            txt.append(f" <{label}>", style="yellow")
    if has_options:
        txt.append(" [OPTIONS]", style="dim cyan")
    if parser.passthrough_hint:
        txt.append(f" {parser.passthrough_hint}", style="dim yellow")

    return txt


class RichArgumentParser(argparse.ArgumentParser):
    """Drop-in `ArgumentParser` subclass with Rich rounded panels and grouped help."""

    def __init__(
        self,
        *args: Any,
        examples: Optional[Sequence[Tuple[str, str]]] = None,
        command_groups: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
        passthrough_hint: Optional[str] = None,
        footer_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.examples: List[Tuple[str, str]] = list(examples) if examples else []
        self.command_groups: List[Tuple[str, List[str]]] = (
            [(title, list(cmds)) for title, cmds in command_groups] if command_groups else []
        )
        self.passthrough_hint: Optional[str] = passthrough_hint
        self.footer_hint: Optional[str] = footer_hint
        super().__init__(*args, **kwargs)

    def _render_help(self, console: Console) -> None:
        console.print()
        console.print(_build_usage_text(self))

        if self.description:
            console.print()
            console.print(Text(self.description.strip(), style="white"))

        # 1. Subcommands (if this parser has subparsers)
        sub_actions = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        if sub_actions:
            all_cmds: Dict[str, Tuple[List[str], str]] = {}
            ordered_names: List[str] = []
            for sa in sub_actions:
                for ca in sa._choices_actions:
                    if ca.help is argparse.SUPPRESS:
                        continue
                    name, aliases, h = _parse_subcommand_entry(ca)
                    all_cmds[name] = (aliases, h)
                    ordered_names.append(name)

            if self.command_groups:
                grouped_seen = set()
                groups_to_render: List[Tuple[str, List[str]]] = []
                for title, names in self.command_groups:
                    present = [n for n in names if n in all_cmds]
                    if present:
                        groups_to_render.append((title, present))
                        grouped_seen.update(present)
                leftover = [n for n in ordered_names if n not in grouped_seen]
                if leftover:
                    groups_to_render.append(("Other Commands", leftover))
            else:
                groups_to_render = [("Commands", ordered_names)]

            max_cmd_width = 0
            for name, (aliases, _) in all_cmds.items():
                label_len = len(name) + (len(f" ({', '.join(aliases)})") if aliases else 0)
                max_cmd_width = max(max_cmd_width, label_len)
            cmd_col_w = min(max(max_cmd_width, 16), 28)

            for title, names in groups_to_render:
                grid = Table.grid(padding=(0, 2), expand=True)
                grid.add_column(width=cmd_col_w, no_wrap=True)
                grid.add_column(ratio=1)
                for name in names:
                    aliases, desc = all_cmds[name]
                    left = Text(name, style="bold cyan")
                    if aliases:
                        left.append(f" ({', '.join(aliases)})", style="dim")
                    grid.add_row(left, Text(desc))
                console.print(
                    Panel(
                        grid,
                        title=f"[bold]{title}[/bold]",
                        title_align="left",
                        box=box.ROUNDED,
                        border_style="dim",
                        padding=(0, 1),
                    )
                )

        # 2. Positional & option groups
        # Compute consistent column widths across all panels for razor-sharp alignment
        option_rows_by_group: List[Tuple[str, List[Tuple[Text, Text, Text]]]] = []
        pos_rows: List[Tuple[Text, Text, Text]] = []

        for group in self._action_groups:
            visible = [
                a
                for a in group._group_actions
                if a.help is not argparse.SUPPRESS and not isinstance(a, argparse._SubParsersAction)
            ]
            if not visible:
                continue

            title = group.title or "Options"
            is_pos_group = title.lower() in _DEFAULT_POS_TITLES
            if title.lower() in _DEFAULT_OPT_TITLES:
                title = "Options"

            rows: List[Tuple[Text, Text, Text]] = []
            for action in visible:
                mv, choices_note = _format_metavar(action)
                help_txt = _format_help_text(action, choices_note)

                if not action.option_strings:
                    name = action.dest
                    is_req = action.nargs not in ("?", "*")
                    prefix = "* " if is_req else "  "
                    left = Text()
                    left.append(prefix, style="bold red" if is_req else "")
                    left.append(name, style="bold cyan")
                    pos_mv = mv if (mv and mv.strip("<>[]") != name) else ""
                    mv_txt = Text(pos_mv, style="yellow") if pos_mv else Text("")
                    pos_rows.append((left, mv_txt, help_txt))
                else:
                    shorts = [s for s in action.option_strings if not s.startswith("--")]
                    longs = [s for s in action.option_strings if s.startswith("--")]
                    left = Text()
                    if shorts:
                        left.append(", ".join(shorts), style="bold green")
                        if longs:
                            left.append(", ", style="dim")
                    if longs:
                        left.append(", ".join(longs), style="bold cyan")
                    mv_txt = Text(mv, style="yellow") if mv else Text("")
                    rows.append((left, mv_txt, help_txt))

            if rows and not is_pos_group:
                option_rows_by_group.append((title, rows))

        # Reorder default "Options" group to the end when custom groups exist,
        # and fold a lone `-h, --help` row into the final custom panel (like Typer/Click).
        if len(option_rows_by_group) > 1:
            default_idx = next(
                (i for i, (t, _) in enumerate(option_rows_by_group) if t == "Options"), None
            )
            if default_idx is not None:
                _, default_rows = option_rows_by_group.pop(default_idx)
                if len(default_rows) == 1 and default_rows[0][0].plain == "-h, --help":
                    option_rows_by_group[-1][1].extend(default_rows)
                else:
                    # Keep `-h, --help` at the very bottom of Options
                    help_rows = [r for r in default_rows if r[0].plain == "-h, --help"]
                    other_rows = [r for r in default_rows if r[0].plain != "-h, --help"]
                    option_rows_by_group.append(("Options", other_rows + help_rows))
        elif len(option_rows_by_group) == 1 and option_rows_by_group[0][0] == "Options":
            rows = option_rows_by_group[0][1]
            help_rows = [r for r in rows if r[0].plain == "-h, --help"]
            other_rows = [r for r in rows if r[0].plain != "-h, --help"]
            option_rows_by_group[0] = ("Options", other_rows + help_rows)

        # Measure shared column widths across all argument/option panels
        all_rows = list(pos_rows)
        for _, grp_rows in option_rows_by_group:
            all_rows.extend(grp_rows)

        if all_rows:
            flag_w = min(max(len(r[0].plain) for r in all_rows), 26)
            mv_max = max((len(r[1].plain) for r in all_rows), default=0)
            mv_w = min(mv_max, 18) if mv_max > 0 else 0

            def _render_panel(title: str, rows: List[Tuple[Text, Text, Text]]) -> None:
                grid = Table.grid(padding=(0, 2), expand=True)
                grid.add_column(width=flag_w, no_wrap=True)
                if mv_w > 0:
                    grid.add_column(width=mv_w, no_wrap=True)
                grid.add_column(ratio=1)
                for left, mv_txt, help_txt in rows:
                    if mv_w > 0:
                        grid.add_row(left, mv_txt, help_txt)
                    else:
                        grid.add_row(left, help_txt)
                console.print(
                    Panel(
                        grid,
                        title=f"[bold]{title}[/bold]",
                        title_align="left",
                        box=box.ROUNDED,
                        border_style="dim",
                        padding=(0, 1),
                    )
                )

            if pos_rows:
                _render_panel("Arguments", pos_rows)
            for title, grp_rows in option_rows_by_group:
                # Skip a standalone "Options" panel containing only `-h, --help` when subcommands exist
                if (
                    sub_actions
                    and title == "Options"
                    and len(grp_rows) == 1
                    and grp_rows[0][0].plain == "-h, --help"
                ):
                    continue
                _render_panel(title, grp_rows)

        # 3. Examples panel
        if self.examples:
            ex_grid = Table.grid(padding=(0, 2), expand=True)
            max_cmd = max(len(cmd) for cmd, _ in self.examples)
            cmd_w = min(max_cmd, max(32, console.width - 34))
            ex_grid.add_column(width=cmd_w, no_wrap=False)
            ex_grid.add_column(ratio=1)
            for cmd, comment in self.examples:
                ex_grid.add_row(Text(cmd, style="green"), Text(f"# {comment}", style="dim"))
            console.print(
                Panel(
                    ex_grid,
                    title="[bold]Examples[/bold]",
                    title_align="left",
                    box=box.ROUNDED,
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        elif self.epilog:
            console.print()
            console.print(Text(self.epilog.strip(), style="dim"))

        if self.footer_hint:
            console.print(Text(f" {self.footer_hint}", style="dim italic"))
        console.print()

    def print_help(self, file: Any = None) -> None:
        target = file if file is not None else sys.stdout
        width = min(Console(file=target).width, 108)
        console = Console(file=target, width=width)
        self._render_help(console)

    def format_help(self) -> str:
        buf = io.StringIO()
        console = Console(file=buf, width=100, force_terminal=False, color_system=None)
        self._render_help(console)
        return buf.getvalue()

    def error(self, message: str) -> None:
        console = Console(file=sys.stderr, width=min(Console(stderr=True).width, 108))
        console.print(_build_usage_text(self))
        console.print(
            Panel(
                Text(message, style="bold red"),
                title="[bold red]Error[/bold red]",
                title_align="left",
                box=box.ROUNDED,
                border_style="red",
                padding=(0, 1),
            )
        )
        console.print(Text(f"Try '{self.prog} --help' for help.", style="dim"))
        self.exit(2)
