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
        console.print(f"  [bold cyan]python3 data/cached_challenge_fineweb.py --variant {variant}[/bold cyan]")
        input("\nPress Enter to return...")
        return

    run_cmd = f"task -G 2 -m 80 bash -c 'export WANDB_ENABLED=1 && export RUN_ID={exp['name']} && export DATA_PATH={data_path} && export TOKENIZER_PATH={token_path} && export VOCAB_SIZE={vocab_size} && torchrun --standalone --nproc_per_node=2 train_gpt.py'"
    
    console.print(f"\n[bold white]Variant Config:[/bold white]")
    console.print(f"  Tokenizer: [cyan]{variant}[/cyan]")
    console.print(f"  Vocab Size: [cyan]{vocab_size}[/cyan]")
    
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
