import pickle
import os
import pandas as pd
import sys
import glob
import plotext as plt
from rich.console import Console
from rich.table import Table

def load_model(path):
    if not os.path.exists(path):
        print(f"Error: Model file '{path}' not found.")
        sys.exit(1)
    
    with open(path, "rb") as f:
        return pickle.load(f)

def get_input(prompt, type_func=str, valid_options=None):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                continue
                
            val = type_func(user_input)
            
            if valid_options and val not in valid_options:
                print(f"Invalid option. Please choose from: {', '.join(map(str, valid_options))}")
                continue
                
            return val
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_func.__name__}.")

def format_currency(val):
    return f"${val:,.0f}"

def collect_user_data():
    print("\n--- Enter Candidate Details ---")
    level = get_input("Level (e.g. E3, E4, E5, E6, E7): ", str, ["E3", "E4", "E5", "E6", "E7"])
    location = get_input("Location (e.g. New York, San Francisco): ", str)
    yoe = get_input("Years of Experience: ", int)
    yac = get_input("Years at Company: ", int)
    
    return pd.DataFrame([{
        "Level": level,
        "Location": location,
        "YearsOfExperience": yoe,
        "YearsAtCompany": yac
    }])

def select_model(console):
    models = glob.glob("*.pkl")
    if not models:
        console.print("[bold red]No model files (*.pkl) found in current directory.[/bold red]")
        console.print("Please run the training CLI first: python3 -m src.cli.train_cli")
        sys.exit(1)
        
    if len(models) == 1:
        console.print(f"[bold blue]Found one model: {models[0]}[/bold blue]")
        return models[0]
        
    console.print("\n[bold]Available Models:[/bold]")
    for i, m in enumerate(models):
        console.print(f"{i+1}. {m}")
        
    while True:
        try:
            choice = int(console.input("\nSelect a model (number): "))
            if 1 <= choice <= len(models):
                return models[choice-1]
            console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a number.[/red]")

def get_ordinal_suffix(n):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def main():
    console = Console()
    console.print("[bold green]Welcome to the Salary Forecasting CLI[/bold green]")
    
    model_path = select_model(console)
    console.print(f"Loading model from: [bold]{model_path}[/bold]")
    model = load_model(model_path)
    
    while True:
        try:
            input_df = collect_user_data()
            
            console.print("\n[bold blue]Calculating prediction...[/bold blue]")
            results = model.predict(input_df)
            
            # Dynamically add columns based on model quantiles
            quantiles = sorted(model.quantiles)
            quantile_labels = [get_ordinal_suffix(int(q * 100)) for q in quantiles]

            # Visualization
            console.print("\n[bold blue]Visualizing Forecast...[/bold blue]")
            plt.clear_figure()
            plt.title("Salary Forecast by Quantile")
            plt.xlabel("Quantile")
            plt.ylabel("Amount ($)")
            plt.theme("pro")
            
            for target, preds in results.items():
                y_values = []
                for q in quantiles:
                    key = f"p{int(q*100)}"
                    y_values.append(preds.get(key, [0])[0])
                
                plt.plot(range(len(quantiles)), y_values, label=target)
            
            plt.xticks(range(len(quantiles)), quantile_labels)
            
            plt.show()

            table = Table(title="Prediction Results")
            table.add_column("Component", style="cyan", no_wrap=True)
            
            for q_label in quantile_labels:
                table.add_column(f"{q_label} Percentile", style="magenta")
            
            for target, preds in results.items():
                row = [target]
                for q in quantiles:
                    key = f"p{int(q*100)}"
                    val = preds.get(key, [0])[0]
                    row.append(format_currency(val))
                table.add_row(*row)
                
            console.print(table)
                
            cont = input("\nForecast another? (y/n): ").strip().lower()
            if cont != 'y':
                console.print("[bold]Goodbye![/bold]")
                break
                
        except KeyboardInterrupt:
            console.print("\n[bold]Goodbye![/bold]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()
