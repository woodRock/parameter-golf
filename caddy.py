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
    
    # Detect Tokenizer Variant
    is_sp8192 = "SP8192" in exp['name']
    variant = "sp8192" if is_sp8192 else "sp1024"
    vocab_size = "8192" if is_sp8192 else "1024"
    model_file = f"fineweb_{vocab_size}_bpe.model"
    
    # Use absolute paths for the run
    data_path = (PROJECT_ROOT / "data" / "datasets" / f"fineweb10B_{variant}").resolve()
    token_path = (PROJECT_ROOT / "data" / "tokenizers" / model_file).resolve()
    
    # Check if data exists
    if not data_path.exists():
        console.print(f"\n[bold red]❌ Error:[/bold red] Data directory not found: {data_path}")
        console.print(f"[yellow]Please run the following on your GPU server first:[/yellow]")
        repo = "kevclark/parameter-golf" if is_sp8192 else "willdepueoai/parameter-golf"
        console.print(f"  [bold cyan]MATCHED_FINEWEB_REPO_ID={repo} python3 data/cached_challenge_fineweb.py --variant {variant} --skip-manifest[/bold cyan]")
        input("\nPress Enter to return...")
        return

    # Detect TTT
    is_ttt = "TTT" in exp['name']
    ttt_flag = "1" if is_ttt else "0"
    
    # Use 1 GPU and 45GB+ capacity requirement to guarantee A40/A6000/L40S (Ampere+)
    num_gpus = 1
    min_mem = 45 # Targets 48GB cards ONLY (Safe for BF16/FlashAttention)
    
    run_cmd = f"task -G {num_gpus} -m {min_mem} bash -c 'export WANDB_ENABLED=1 && export TTT_ENABLED={ttt_flag} && export MAX_WALLCLOCK_SECONDS=4800 && export RUN_ID={exp['name']} && export DATA_PATH={data_path} && export TOKENIZER_PATH={token_path} && export VOCAB_SIZE={vocab_size} && torchrun --standalone --nproc_per_node={num_gpus} train_gpt.py'"
    
    console.print(f"\n[bold white]Variant Config:[/bold white]")
    console.print(f"  Tokenizer: [cyan]{variant}[/cyan]  Vocab: [cyan]{vocab_size}[/cyan]  TTT: [cyan]{'Enabled' if is_ttt else 'Disabled'}[/cyan]")
    console.print(f"  Hardware: [cyan]{num_gpus} GPU[/cyan]  Min-Mem: [cyan]{min_mem}GB[/cyan]")
    
    console.print(f"\n[bold white]Command:[/bold white]\n  {run_cmd}")
    
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
