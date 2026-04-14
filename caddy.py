#!/usr/bin/env python3
import os
import re
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Label, Static, TabbedContent, TabPane, Button
from textual.containers import Horizontal, Vertical, Container
from textual.binding import Binding
from textual.reactive import reactive
from textual.timer import Timer
from textual import on

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Logic Functions ---

def get_leaderboard():
    readme_path = PROJECT_ROOT / "README.md"
    if not readme_path.exists(): return []
    content = readme_path.read_text()
    table_match = re.search(r"## Leaderboard\n\n(.*?)\n\n", content, re.DOTALL)
    if not table_match: return []
    rows = []
    lines = table_match.group(1).split("\n")
    for line in lines[2:]:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2: rows.append(parts)
    return rows[:50]

def get_bpb_from_logs(exp_path):
    sub_json = exp_path / "submission.json"
    if sub_json.exists():
        try: return json.loads(sub_json.read_text()).get("val_bpb", "N/A")
        except: pass
    log_files = []
    exp_logs_dir = exp_path / "logs"
    if exp_logs_dir.exists():
        log_files = [f for f in exp_logs_dir.iterdir() if f.is_file() and f.suffix in (".txt", ".log")]
    global_log = PROJECT_ROOT / "logs" / f"{exp_path.name}.txt"
    if global_log.exists(): log_files.append(global_log)
    if not log_files: return "N/A"
    all_matches = []
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()
            matches = re.findall(r"val_bpb[:=\s]+(\d+\.\d+)", content)
            if matches: all_matches.append(matches[-1])
        except: pass
    return all_matches[-1] if all_matches else "N/A"

def is_my_experiment(path):
    try:
        res = subprocess.run(
            ["git", "log", "--reverse", "--format=%an", str(path)],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        authors = res.stdout.strip().split("\n")
        return authors and "woodRock" in authors[0]
    except: return False

def list_experiments():
    records_dir = PROJECT_ROOT / "records"
    if not records_dir.exists(): return [], []
    my_exps, others = [], []
    today = datetime.now().strftime("%Y-%m-%d")
    for track in records_dir.iterdir():
        if not track.is_dir(): continue
        for d in track.iterdir():
            if not d.is_dir(): continue
            info = {"name": d.name, "path": d, "bpb": get_bpb_from_logs(d)}
            if d.name.startswith(today) or is_my_experiment(d): my_exps.append(info)
            else: others.append(info)
    my_exps.sort(key=lambda x: x["name"], reverse=True)
    others.sort(key=lambda x: x["name"], reverse=True)
    return my_exps, others

def get_running_tasks() -> List[Dict]:
    try:
        res = subprocess.run(["task", "-l"], capture_output=True, text=True)
        lines = res.stdout.strip().split("\n")
        if len(lines) < 3: return []
        tasks = []
        # ID  STATUS      G  SERVER           GPU  SUBMITTED     RUNTIME  LABEL / COMMAND
        for line in lines[2:]:
            if "task(s)" in line: break
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 6:
                tasks.append({
                    "id": parts[0],
                    "status": parts[1],
                    "g": parts[2],
                    "submitted": parts[5],
                    "runtime": parts[6] if len(parts) > 6 else "-",
                    "label": parts[7] if len(parts) > 7 else "-"
                })
        return tasks
    except: return []

# --- TUI App ---

class CaddyApp(App):
    CSS = """
    Screen {
        background: $surface;
    }

    #header-title {
        background: $accent;
        color: white;
        text-style: bold;
        padding: 0 1;
        width: 100%;
        text-align: center;
        height: 1;
    }

    DataTable {
        height: 1fr;
        border: none;
    }

    .title-label {
        width: 100%;
        text-align: center;
        background: $primary;
        color: white;
        text-style: bold;
        padding: 0 1;
    }

    #left-pane {
        width: 50%;
        height: 100%;
        border-right: solid $primary;
    }

    #right-pane {
        width: 50%;
        height: 100%;
    }

    TabPane {
        padding: 0;
    }

    Footer {
        background: $surface;
        color: $text;
    }

    .status-running {
        color: $success;
    }

    .status-queued {
        color: $warning;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh All"),
        Binding("g", "toggle_global", "Toggle Global Exps"),
        Binding("enter", "launch", "Launch Selected", show=True),
        Binding("t", "toggle_theme", "Toggle Light/Dark"),
        Binding("left", "prev_tab", "Prev Tab", show=False),
        Binding("right", "next_tab", "Next Tab", show=False),
    ]

    show_global = reactive(False)

    def __init__(self):
        super().__init__()
        self.my_exps = []
        self.others = []
        self.leaderboard = []
        self.tasks = []
        self.refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(f"⛳ GOLF CADDY | {PROJECT_ROOT}", id="header-title")
        with Horizontal():
            with Vertical(id="left-pane"):
                with TabbedContent(id="tabs"):
                    with TabPane("🧪 Experiments", id="tab-exps"):
                        yield Label("Your Experiments", id="exp-title", classes="title-label")
                        yield DataTable(id="exp-table")
                    with TabPane("📡 Active Tasks", id="tab-tasks"):
                        yield Label("Running & Queued Tasks", classes="title-label")
                        yield DataTable(id="task-table")
            with Vertical(id="right-pane"):
                yield Label("🏆 Global Leaderboard", classes="title-label")
                yield DataTable(id="leaderboard-table")
        yield Footer()

    def on_mount(self) -> None:
        self.setup_tables()
        self.action_refresh()
        self.query_one("#exp-table").focus()
        self.refresh_timer = self.set_interval(5, self.refresh_tasks)

    @on(TabbedContent.TabActivated)
    def on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "tab-exps":
            self.query_one("#exp-table").focus()
        elif event.pane.id == "tab-tasks":
            self.query_one("#task-table").focus()

    def action_next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        if tabs.active == "tab-exps":
            tabs.active = "tab-tasks"
        else:
            tabs.active = "tab-exps"

    def action_prev_tab(self) -> None:
        self.action_next_tab()

    def setup_tables(self):
        exp_table = self.query_one("#exp-table", DataTable)
        exp_table.cursor_type = "row"
        exp_table.add_columns("BPB", "Experiment ID")
        
        task_table = self.query_one("#task-table", DataTable)
        task_table.cursor_type = "row"
        task_table.add_columns("ID", "Status", "Runtime", "Label")
        
        lb_table = self.query_one("#leaderboard-table", DataTable)
        lb_table.cursor_type = "row"
        lb_table.add_columns("BPB", "Run")

    def action_refresh(self) -> None:
        self.my_exps, self.others = list_experiments()
        self.leaderboard = get_leaderboard()
        self.refresh_tasks()
        self.update_tables()

    def refresh_tasks(self) -> None:
        self.tasks = get_running_tasks()
        task_table = self.query_one("#task-table", DataTable)
        task_table.clear()
        for t in self.tasks:
            status_style = "[bold green]" if t["status"] == "running" else "[yellow]"
            task_table.add_row(
                t["id"],
                f"{status_style}{t['status']}[/]",
                t["runtime"],
                t["label"]
            )

    def update_tables(self) -> None:
        target_list = self.others if self.show_global else self.my_exps
        title = "🌍 All Competition Records" if self.show_global else "🧪 Your Experiments"
        self.query_one("#exp-title", Label).update(title)
        
        exp_table = self.query_one("#exp-table", DataTable)
        exp_table.clear()
        for exp in target_list:
            exp_table.add_row(str(exp["bpb"]), exp["name"])

        lb_table = self.query_one("#leaderboard-table", DataTable)
        lb_table.clear()
        for row in self.leaderboard:
            lb_table.add_row(row[1], row[0])

    def action_toggle_global(self) -> None:
        self.show_global = not self.show_global
        self.update_tables()

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def action_launch(self) -> None:
        active_tab = self.query_one(TabbedContent).active
        if active_tab == "tab-exps":
            exp_table = self.query_one("#exp-table", DataTable)
            cursor_row = exp_table.cursor_row
            if cursor_row is not None:
                target_list = self.others if self.show_global else self.my_exps
                if cursor_row < len(target_list):
                    self.launch_experiment(target_list[cursor_row])
        elif active_tab == "tab-tasks":
            # Potentially add task cancellation or logs here
            self.notify("Task management coming soon!", severity="info")

    def launch_experiment(self, exp):
        is_sp8192 = "SP8192" in exp['name']
        variant = "sp8192" if is_sp8192 else "sp1024"
        vocab_size = "8192" if is_sp8192 else "1024"
        model_file = f"fineweb_{vocab_size}_bpe.model"
        data_path = (PROJECT_ROOT / "data" / "datasets" / f"fineweb10B_{variant}").resolve()
        token_path = (PROJECT_ROOT / "data" / "tokenizers" / model_file).resolve()
        
        if not data_path.exists():
            self.notify(f"Data directory not found: {data_path}", severity="error")
            return

        is_ttt = "TTT" in exp['name']
        ttt_flag = "1" if is_ttt else "0"
        
        # Original script logic for experiment launching
        run_cmd = f"bash -c 'export WANDB_ENABLED=1 && export TTT_ENABLED={ttt_flag} && export MAX_WALLCLOCK_SECONDS=4800 && export RUN_ID={exp['name']} && export DATA_PATH={data_path} && export TOKENIZER_PATH={token_path} && export VOCAB_SIZE={vocab_size} && torchrun --standalone --nproc_per_node=1 train_gpt.py --wallclock 4800'"
        
        # Wrap in task command as requested: task -G 1 -m 45 -n <label> <script>
        task_cmd = f"task -G 1 -m 45 -n {exp['name']} {run_cmd}"
        
        def run_it():
            os.system("clear")
            print(f"🚀 Spooling experiment to task queue: {exp['name']}\n")
            print(f"Command:\n{task_cmd}\n")
            subprocess.run(f"cd {exp['path']} && {task_cmd}", shell=True)
            print("\n[bold green]✅ Task submitted to queue.[/bold green]")
            input("\nPress Enter to return to Caddy...")

        self.app.suspend_process(run_it)
        self.action_refresh()

if __name__ == "__main__":
    CaddyApp().run()
