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
    
    # Try to find the latest log file in the experiment dir
    log_files = list(exp_path.glob("*.txt")) + list(exp_path.glob("*.log"))
    if not log_files:
        return "N/A"
    
    # Sort by mtime to get the latest
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    try:
        content = latest_log.read_text()
        # Look for the last final_int8_zlib_roundtrip or val_bpb entry
        matches = re.findall(r"val_bpb:(\d+\.\d+)", content)
        if matches:
            return matches[-1]
    except: pass
    return "N/A"

def is_my_experiment(path):
    try:
        # Check if the directory was created/touched by woodRock in git
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
    
    for track in records_dir.iterdir():
        if track.is_dir():
            for d in track.iterdir():
                if d.is_dir():
                    info = {"name": d.name, "path": d, "bpb": get_bpb_from_logs(d)}
                    if is_my_experiment(d) or "2026-04-14" in d.name:
                        my_exps.append(info)
                    else:
                        others.append(info)
    
    my_exps.sort(key=lambda x: x["name"], reverse=True)
    others.sort(key=lambda x: x["name"], reverse=True)
    return my_exps, others

def show_main_menu():
    console.clear()
    console.print(f"[bold green]⛳ GOLF CADDY[/bold green] [dim]| {PROJECT_ROOT}[/dim]", justify="center")
    
    my_exps, others = list_experiments()
    
    # Create side-by-side or stacked tables
    table = Table(box=None, padding=(0, 2))
    table.add_column("Your Experiments", style="bold green")
    table.add_column("Global SOTA", style="bold yellow")
    
    # Local Table
    local_t = Table(border_style="green", header_style="bold green", x_ratio=None)
    local_t.add_column("#", style="dim", width=2)
    local_t.add_column("Experiment", width=30)
    local_t.add_column("BPB", justify="right")
    for i, exp in enumerate(my_exps[:15]):
        local_t.add_row(str(i+1), exp["name"], str(exp["bpb"]))
        
    # SOTA Table (from README)
    sota_t = Table(border_style="yellow", header_style="bold yellow")
    sota_t.add_column("Run", width=30)
    sota_t.add_column("BPB", justify="right")
    lb = get_leaderboard()
    for row in lb[:10]:
        sota_t.add_row(row[0][:30], row[1])
        
    console.print(local_t, sota_t, justify="center")
    console.print("\n [bold cyan][1-N][/bold cyan] Launch   [bold yellow][R][/bold yellow] Refresh   [bold red][Q][/bold red] Quit", justify="center")

def launch_experiment(exp):
    console.print(Panel(f"🚀 Preparing to launch: [bold green]{exp['name']}[/bold green]"))
    
    # Use absolute paths for the run
    data_path = (PROJECT_ROOT / "data" / "datasets" / "fineweb10B_sp1024").resolve()
    token_path = (PROJECT_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()
    vocab_size = "1024"
    
    # Create the command
    run_cmd = (
        f"ts -G 2 torchrun --standalone --nproc_per_node=2 train_gpt.py"
    )
    
    env_vars = {
        "RUN_ID": exp['name'],
        "WANDB_ENABLED": "1",
        "DATA_PATH": str(data_path),
        "TOKENIZER_PATH": str(token_path),
        "VOCAB_SIZE": vocab_size
    }
    
    console.print("\n[bold white]Environment Variables:[/bold white]")
    for k, v in env_vars.items():
        console.print(f"  {k}={v}")
    
    console.print(f"\n[bold white]Command:[/bold white]\n  {run_cmd}")
    
    confirm = Prompt.ask("\nLaunch this task?", choices=["y", "n"], default="y")
    if confirm == "y":
        # Execute in the directory of the experiment
        env_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
        full_cmd = f"cd {exp['path']} && {env_str} {run_cmd}"
        
        try:
            subprocess.run(full_cmd, shell=True, check=True)
            console.print("\n[bold green]✅ Task Spooled![/bold green] (Check `ts` for status)")
        except Exception as e:
            console.print(f"\n[bold red]❌ Failed to spool task:[/bold red] {e}")
    
    input("\nPress Enter to return to menu...")

def main():
    while True:
        show_main_menu()
        choice = Prompt.ask("\n[bold cyan]Choose an action[/bold cyan]").lower()
        
        if choice == 'q':
            break
        elif choice == 'r':
            continue
        elif choice.isdigit():
            idx = int(choice) - 1
            my_exps, others = list_experiments()
            if 0 <= idx < len(my_exps):
                launch_experiment(my_exps[idx])
            else:
                console.print("[bold red]Invalid index![/bold red]")
                time.sleep(1)
        else:
            console.print("[bold red]Unknown command![/bold red]")

if __name__ == "__main__":
    main()
