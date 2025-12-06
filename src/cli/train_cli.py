import os
from rich.console import Console
from src.model.train import train_model

def get_input(console, prompt, default=None):
    prompt_str = f"{prompt} [default: {default}]: " if default else f"{prompt}: "
    user_input = console.input(prompt_str).strip()
    return user_input if user_input else default

def main():
    console = Console()
    console.print("[bold green]Salary Forecasting Training CLI[/bold green]")
    
    csv_path = get_input(console, "Input CSV path", "salaries-list.csv")
    config_path = get_input(console, "Config JSON path", "config.json")
    output_path = get_input(console, "Output model path", "salary_model.pkl")
    
    if not os.path.exists(csv_path):
        console.print(f"[bold red]Error: Input file '{csv_path}' not found.[/bold red]")
        return
        
    if not os.path.exists(config_path):
        console.print(f"[bold red]Error: Config file '{config_path}' not found.[/bold red]")
        return
        
    console.print(f"\n[bold blue]Starting training...[/bold blue]")
    try:
        train_model(csv_path, config_path, output_path)
        console.print(f"\n[bold green]Training completed successfully! Model saved to {output_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")

if __name__ == "__main__":
    main()
