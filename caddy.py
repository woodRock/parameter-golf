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
        width: 100%;
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

    TabPane {
        padding: 0;
        height: 1fr;
    }

    TabbedContent {
        height: 1fr;
    }

    Footer {
        background: $surface;
        color: $text;
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
        with TabbedContent(id="tabs"):
            with TabPane("🧪 Experiments", id="tab-exps"):
                yield Label("Your Experiments", id="exp-title", classes="title-label")
                yield DataTable(id="exp-table")
            with TabPane("📡 Active Tasks", id="tab-tasks"):
                yield Label("Running & Queued Tasks", classes="title-label")
                yield DataTable(id="task-table")
            with TabPane("🏆 Leaderboard", id="tab-leaderboard"):
                yield Label("Global Leaderboard", classes="title-label")
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
        elif event.pane.id == "tab-leaderboard":
            self.query_one("#leaderboard-table").focus()

    def action_next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        tab_ids = ["tab-exps", "tab-tasks", "tab-leaderboard"]
        try:
            curr_idx = tab_ids.index(tabs.active)
            next_idx = (curr_idx + 1) % len(tab_ids)
            tabs.active = tab_ids[next_idx]
        except ValueError:
            tabs.active = "tab-exps"

    def action_prev_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        tab_ids = ["tab-exps", "tab-tasks", "tab-leaderboard"]
        try:
            curr_idx = tab_ids.index(tabs.active)
            prev_idx = (curr_idx - 1) % len(tab_ids)
            tabs.active = tab_ids[prev_idx]
        except ValueError:
            tabs.active = "tab-exps"

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
        try:
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
        except: pass

    def update_tables(self) -> None:
        target_list = self.others if self.show_global else self.my_exps
        title = "🌍 All Competition Records" if self.show_global else "🧪 Your Experiments"
        try:
            self.query_one("#exp-title", Label).update(title)
            
            exp_table = self.query_one("#exp-table", DataTable)
            exp_table.clear()
            for exp in target_list:
                exp_table.add_row(str(exp["bpb"]), exp["name"])

            lb_table = self.query_one("#leaderboard-table", DataTable)
            lb_table.clear()
            for row in self.leaderboard:
                lb_table.add_row(row[1], row[0])
        except: pass

    def action_toggle_global(self) -> None:
        self.show_global = not self.show_global
        self.update_tables()

    def action_toggle_theme(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        # If we are on the experiments table, trigger the launch action
        if event.data_table.id == "exp-table":
            self.action_launch()

    def action_launch(self) -> None:
        active_tab = self.query_one(TabbedContent).active
        if active_tab == "tab-exps":
            exp_table = self.query_one("#exp-table", DataTable)
            cursor_row = exp_table.cursor_row
            if cursor_row is not None:
                # Get the experiment name from the second column (index 1) of the highlighted row
                row_data = exp_table.get_row_at(cursor_row)
                exp_id = row_data[1]
                
                target_list = self.others if self.show_global else self.my_exps
                # Find the experiment in the metadata list by name
                exp = next((e for e in target_list if e["name"] == exp_id), None)
                
                if exp:
                    self.launch_experiment(exp)
        elif active_tab == "tab-tasks":
            self.notify("Task management coming soon!", severity="info")
        elif active_tab == "tab-leaderboard":
            self.notify("Leaderboard is read-only.", severity="info")

    def launch_experiment(self, exp):
        # Case-insensitive check for SP8192 or specific known models like "Bride" or "DeepSeek"
        exp_name_upper = exp['name'].upper()
        # Explicit check for "The_Bride", "BRIDE", or "DEEPSEEK" as these should always be 8192
        is_sota = any(k in exp_name_upper for k in ["SP8192", "8192", "BRIDE", "DEEPSEEK"])
        
        variant = "sp8192" if is_sota else "sp1024"
        vocab_size = "8192" if is_sota else "1024"
        model_file = f"fineweb_{vocab_size}_bpe.model"
        
        # Determine number of GPUs to use (forced to 1 for ECS server)
        nproc = os.environ.get("NPROC", "1")
        
        # When running from the experiment dir, data/token paths need to be relative to it
        # or absolute. Using absolute paths from PROJECT_ROOT is safest.
        data_path = (PROJECT_ROOT / "data" / "datasets" / f"fineweb10B_{variant}").resolve()
        token_path = (PROJECT_ROOT / "data" / "tokenizers" / model_file).resolve()
        
        if not data_path.exists():
            self.notify(f"Data directory not found: {data_path}", severity="error")
            return

        is_ttt = "TTT" in exp_name_upper
        ttt_flag = "1" if is_ttt else "0"
        
        # The script is in the current directory once we cd
        script_name = "train_gpt.py"
        if not (exp['path'] / script_name).exists():
            script_path = (PROJECT_ROOT / script_name).resolve()
        else:
            script_path = script_name

        # Build the command exactly as it would be typed in a shell
        env_str = (
            f"WANDB_ENABLED=1 TTT_ENABLED={ttt_flag} RUN_ID={exp['name']} "
            f"DATA_PATH={data_path} TOKENIZER_PATH={token_path} VOCAB_SIZE={vocab_size}"
        )
        run_cmd = f"task -G {nproc} -m 45 -n {exp['name']} {env_str} torchrun --standalone --nproc_per_node={nproc} {script_path}"
        
        def run_it(interactive: bool = True):
            if interactive:
                os.system("clear")
                print(f"🚀 Launching experiment: {exp['name']}\n")
                print(f"Command:\n{run_cmd}\n")
            
            # Execute from the experiment directory
            subprocess.run(f"cd {exp['path']} && {run_cmd}", shell=True)
            
            if interactive:
                print("\n[bold green]✅ Task submitted to queue.[/bold green]")
                print("\nPress Enter to return to Caddy...")
                sys.stdin.readline()

        # suspend_process is available in Textual 0.49.0+
        if hasattr(self, "suspend_process"):
            self.suspend_process(run_it)
        else:
            # Fallback for older Textual: run without blocking input to avoid hang
            run_it(interactive=False)
            self.notify(f"🚀 Queued: {exp['name']}", severity="info")
        
        self.action_refresh()

if __name__ == "__main__":
    CaddyApp().run()
