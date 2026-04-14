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

# Find the project root (where this script lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

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

def list_experiments():
    base_dir = PROJECT_ROOT / "records" / "track_10min_16mb"
    if not base_dir.exists():
        return []
    
    experiments = []
    for d in sorted(base_dir.iterdir(), reverse=True):
        if d.is_dir():
            sub_json = d / "submission.json"
            bpb = "N/A"
            if sub_json.exists():
                try:
                    data = json.loads(sub_json.read_text())
                    bpb = data.get("val_bpb", "N/A")
                except:
                    pass
            experiments.append({"name": d.name, "path": d, "bpb": bpb})
    return experiments

def show_main_menu():
    console.clear()
    console.print(Panel("[bold green]⛳ GOLF CADDY[/bold green]\n[dim]The Parameter Golf Experiment Manager[/dim]", expand=False))
    
    # Show Leaderboard
    lb = get_leaderboard()
    if lb:
        table = Table(title="🏆 Global Leaderboard (Top 10)", border_style="yellow")
        table.add_column("Run", style="cyan", no_wrap=True)
        table.add_column("Score (BPB)", style="green")
        table.add_column("Author", style="magenta")
        for row in lb:
            table.add_row(row[0], row[1], row[2])
        console.print(table)

    # Show Local Experiments
    exps = list_experiments()
    table = Table(title="🧪 Your Experiments", border_style="blue")
    table.add_column("#", style="dim")
    table.add_column("Experiment ID", style="bold white")
    table.add_column("Last BPB", style="green")
    
    for i, exp in enumerate(exps):
        table.add_row(str(i+1), exp["name"], str(exp["bpb"]))
    console.print(table)

    console.print("\n[bold cyan][1-N][/bold cyan] Launch Experiment   [bold yellow][R][/bold yellow] Refresh   [bold red][Q][/bold red] Quit")

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
            exps = list_experiments()
            if 0 <= idx < len(exps):
                launch_experiment(exps[idx])
            else:
                console.print("[bold red]Invalid index![/bold red]")
                time.sleep(1)
        else:
            console.print("[bold red]Unknown command![/bold red]")

if __name__ == "__main__":
    main()
