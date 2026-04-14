#!/usr/bin/env python3
import os
import re
import json
import time
import subprocess
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt

console = Console()

# Find the project root (follow symlinks to the actual script location)
PROJECT_ROOT = Path(__file__).resolve().parent

def get_leaderboard():
    readme_path = PROJECT_ROOT / "README.md"
    if not readme_path.exists():
        return []
    
    content = readme_path.read_text()
    # Extract the Leaderboard table using regex
    table_match = re.search(r"## Leaderboard\n\n(.*?)\n\n", content, re.DOTALL)
    if not table_match:
        return []
    
    rows = []
    lines = table_match.group(1).split("\n")
    for line in lines[2:]: # Skip header and separator
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3:
            rows.append(parts)
    return rows[:10]

def get_bpb_from_logs(exp_path):
    # Try submission.json first
    sub_json = exp_path / "submission.json"
    if sub_json.exists():
        try:
            return json.loads(sub_json.read_text()).get("val_bpb", "N/A")
        except: pass

    # Check for a 'logs' subdirectory within the experiment (primary location)
    log_files = []
    exp_logs_dir = exp_path / "logs"
    if exp_logs_dir.exists():
        log_files = [f for f in exp_logs_dir.iterdir() if f.is_file() and f.suffix in (".txt", ".log")]

    # Also check the global logs folder for a file named after the experiment
    global_log = PROJECT_ROOT / "logs" / f"{exp_path.name}.txt"
    if global_log.exists():
        log_files.append(global_log)

    if not log_files:
        return "N/A"

    # Scan all log files and collect val_bpb entries
    all_matches = []
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()

            matches = re.findall(r"val_bpb[:=\s]+(\d+\.\d+)", content)
            if matches:
                all_matches.append(matches[-1])
        except:
            pass

    if all_matches:
        return all_matches[-1]
    
    return "N/A"

def is_my_experiment(path):
    try:
        res = subprocess.run(
            ["git", "log", "--format=%an", "-n", "1", "--", str(path)],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        return "woodRock" in res.stdout
    except:
        return False

def list_experiments():
    records_dir = PROJECT_ROOT / "records"
    if not records_dir.exists():
        return [], []
    
    my_exps = []
    others = []
    today = "2026-04-14"
    
    for track in records_dir.iterdir():
        if track.is_dir():
            for d in track.iterdir():
                if d.is_dir():
                    info = {"name": d.name, "path": d, "bpb": get_bpb_from_logs(d)}
                    if d.name.startswith(today) or is_my_experiment(d):
                        my_exps.append(info)
                    else:
                        others.append(info)
    
    my_exps.sort(key=lambda x: x["name"], reverse=True)
    others.sort(key=lambda x: x["name"], reverse=True)
    return my_exps, others

def show_main_menu(show_global=False):
    console.clear()
    console.print(f"[bold green]⛳ GOLF CADDY[/bold green] [dim]| {PROJECT_ROOT}[/dim]", justify="center")
    
    my_exps, others = list_experiments()
    
    if not show_global:
        # User View
        table = Table(title="🧪 Your Experiments", border_style="green", header_style="bold green")
        table.add_column("#", style="dim", width=2)
        table.add_column("Experiment ID", width=50)
        table.add_column("Latest BPB", justify="right")
        
        for i, exp in enumerate(my_exps):
            table.add_row(str(i+1), exp["name"], str(exp["bpb"]))
        console.print(table, justify="center")
        
        # Mini Leaderboard
        lb_table = Table(title="🏆 Top 5 to Beat", border_style="yellow")
        lb_table.add_column("Run", width=40)
        lb_table.add_column("BPB", justify="right")
        lb = get_leaderboard()
        for row in lb[:5]:
            lb_table.add_row(row[0], row[1])
        console.print(lb_table, justify="center")
        
        console.print("\n [bold cyan][1-N][/bold cyan] Launch   [bold yellow][G][/bold yellow] Show All Records   [bold yellow][R][/bold yellow] Refresh   [bold red][Q][/bold red] Quit", justify="center")
    else:
        # Global View
        table = Table(title="🌍 All Competition Records", border_style="blue", header_style="bold blue")
        table.add_column("#", style="dim", width=2)
        table.add_column("Record ID", width=50)
        table.add_column("BPB", justify="right")
        
        for i, exp in enumerate(others):
            table.add_row(str(i+1), exp["name"], str(exp["bpb"]))
        console.print(table, justify="center")
        console.print("\n [bold yellow][B][/bold yellow] Back to My Experiments   [bold red][Q][/bold red] Quit", justify="center")

def launch_experiment(exp):
    console.print(Panel(f"🚀 Preparing to launch: [bold green]{exp['name']}[/bold green]"))

    # Use absolute paths for the run
    data_path = (PROJECT_ROOT / "data" / "datasets" / "fineweb10B_sp1024").resolve()
    token_path = (PROJECT_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()
    vocab_size = "1024"

    env_vars = {
        "RUN_ID": exp['name'],
        "WANDB_ENABLED": "1",
        "DATA_PATH": str(data_path),
        "TOKENIZER_PATH": str(token_path),
        "VOCAB_SIZE": vocab_size
    }
    
    export_str = " && ".join([f"export {k}={v}" for k, v in env_vars.items()])
    inner_cmd = f"{export_str} && torchrun --standalone --nproc_per_node=2 train_gpt.py"
    run_cmd = f"task -G 2 bash -c '{inner_cmd}'"
    
    console.print("\n[bold white]Environment Variables:[/bold white]")
    for k, v in env_vars.items():
        console.print(f"  {k}={v}")

    console.print(f"\n[bold white]Command:[/bold white]\n  task -G 2 bash -c 'export ... && torchrun ...'")

    confirm = Prompt.ask("\nLaunch this task?", choices=["y", "n"], default="y")
    if confirm == "y":
        full_cmd = f"cd {exp['path']} && {run_cmd}"
        
        try:
            subprocess.run(full_cmd, shell=True, check=True)
            console.print("\n[bold green]✅ Task Spooled![/bold green]")
        except Exception as e:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
    
    input("\nPress Enter to return...")

def main():
    show_global = False
    while True:
        show_main_menu(show_global)
        choice = Prompt.ask("\n[bold cyan]Action[/bold cyan]").lower()
        
        if choice == 'q':
            break
        elif choice == 'r':
            continue
        elif choice == 'g':
            show_global = True
        elif choice == 'b':
            show_global = False
        elif choice.isdigit():
            idx = int(choice) - 1
            my_exps, others = list_experiments()
            target_list = others if show_global else my_exps
            if 0 <= idx < len(target_list):
                launch_experiment(target_list[idx])
            else:
                console.print("[bold red]Invalid index![/bold red]")
                time.sleep(1)
        else:
            console.print("[bold red]Unknown command![/bold red]")

if __name__ == "__main__":
    main()
