#!/usr/bin/env python3
"""
PoMA Frameworks Comparison - Main Entry Point
============================================

This is the main entry point for the comprehensive mediation analysis
framework comparison project. It provides a menu-driven interface to:

1. Run step-by-step demonstrations
2. Generate comprehensive comparisons
3. Analyze edge cases
4. Use the diagnostic toolkit
5. View mathematical foundations

Usage:
    python main.py
"""

import os
import sys
import subprocess
from typing import List, Tuple


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the main header."""
    print("="*80)
    print("MEDIATION ANALYSIS FRAMEWORKS COMPARISON")
    print("="*80)
    print("\nA comprehensive toolkit for understanding and comparing:")
    print("- Traditional Baron & Kenny approach")
    print("- Frisch-Waugh-Lovell (FWL) theorem")
    print("- Double Machine Learning (DML)")
    print("- Causal mediation (Natural Effects)")
    print()


def print_menu():
    """Print the main menu."""
    print("\nMAIN MENU")
    print("---------")
    print("1. Step-by-step demonstrations")
    print("2. Comprehensive comparison (all methods, all scenarios)")
    print("3. Edge cases analysis")
    print("4. Diagnostic toolkit (analyze your own data)")
    print("5. Mathematical foundations")
    print("6. View project documentation")
    print("7. Run all demonstrations")
    print("0. Exit")
    print()


def run_step_demos():
    """Run step-by-step demonstrations."""
    clear_screen()
    print("STEP-BY-STEP DEMONSTRATIONS")
    print("="*80)
    print("\nChoose a demonstration:")
    print("1. Linear equivalence (Traditional = FWL = DML)")
    print("2. Non-linear robustness (DML with ML)")
    print("3. Interactions (Natural effects)")
    print("4. Run all three in sequence")
    print("0. Back to main menu")
    
    choice = input("\nEnter your choice: ").strip()
    
    demos = {
        '1': 'demonstrations/step1_linear_equivalence.py',
        '2': 'demonstrations/step2_nonlinear_dml.py',
        '3': 'demonstrations/step3_interactions_causal.py'
    }
    
    if choice in demos:
        print(f"\nRunning {demos[choice]}...")
        subprocess.run([sys.executable, demos[choice]])
        input("\nPress Enter to continue...")
    elif choice == '4':
        for demo in demos.values():
            print(f"\nRunning {demo}...")
            subprocess.run([sys.executable, demo])
            input("\nPress Enter for next demo...")
    elif choice != '0':
        print("Invalid choice!")
        input("Press Enter to continue...")


def run_comprehensive():
    """Run comprehensive comparison."""
    clear_screen()
    print("COMPREHENSIVE COMPARISON")
    print("="*80)
    print("\nThis will run all 6 methods on all 11 data scenarios.")
    print("It generates:")
    print("- Detailed results CSV")
    print("- Performance heatmaps")
    print("- Method recommendations")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == 'y':
        print("\nRunning comprehensive comparison...")
        subprocess.run([sys.executable, 'demonstrations/comprehensive_comparison.py'])
        
        print("\nResults saved to:")
        print("- outputs/tables/comprehensive_results.csv")
        print("- outputs/figures/performance_heatmap.png")
        input("\nPress Enter to continue...")


def run_edge_cases():
    """Run edge cases analysis."""
    clear_screen()
    print("EDGE CASES ANALYSIS")
    print("="*80)
    print("\nAnalyzing challenging scenarios:")
    print("- Symmetric relationships")
    print("- Suppression effects")
    print("- Near-zero effects")
    print("- Inconsistent mediation")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == 'y':
        print("\nRunning edge cases analysis...")
        subprocess.run([sys.executable, 'demonstrations/edge_cases_demo.py'])
        input("\nPress Enter to continue...")


def run_diagnostic():
    """Run diagnostic toolkit."""
    clear_screen()
    print("DIAGNOSTIC TOOLKIT")
    print("="*80)
    print("\nThe diagnostic toolkit helps you:")
    print("- Check assumptions for your data")
    print("- Get method recommendations")
    print("- Visualize mediation relationships")
    
    print("\nTo use with your own data:")
    print("```python")
    print("from tools.diagnostic_toolkit import diagnose_mediation_data")
    print("report = diagnose_mediation_data(X, M, Y)")
    print("report.show()")
    print("```")
    
    demo = input("\nRun example demonstration? (y/n): ").strip().lower()
    if demo == 'y':
        subprocess.run([sys.executable, 'tools/diagnostic_toolkit.py'])
        input("\nPress Enter to continue...")


def view_math():
    """View mathematical foundations."""
    clear_screen()
    print("MATHEMATICAL FOUNDATIONS")
    print("="*80)
    print("\nThe mathematical proofs notebook covers:")
    print("- Equivalence of Traditional and FWL")
    print("- DML reduction formula derivation")
    print("- Natural effects decomposition")
    print("- When CDE = NDE")
    
    view = input("\nView mathematical proofs? (y/n): ").strip().lower()
    if view == 'y':
        subprocess.run([sys.executable, 'notebooks/mathematical_proofs.py'])
        input("\nPress Enter to continue...")


def view_docs():
    """View project documentation."""
    clear_screen()
    print("PROJECT DOCUMENTATION")
    print("="*80)
    
    docs = [
        ("README.md", "Project overview and usage"),
        ("MASTER_CONTEXT.md", "Detailed implementation plan"),
        ("docs/theoretical_background.md", "Theoretical foundations"),
        ("docs/implementation_notes.md", "Technical details")
    ]
    
    print("\nAvailable documentation:")
    for i, (file, desc) in enumerate(docs, 1):
        print(f"{i}. {file} - {desc}")
    print("0. Back to main menu")
    
    choice = input("\nEnter your choice: ").strip()
    
    if choice.isdigit() and 0 < int(choice) <= len(docs):
        file = docs[int(choice)-1][0]
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = f.read()
            
            # Simple pager
            lines = content.split('\n')
            page_size = 25
            start = 0
            
            while start < len(lines):
                clear_screen()
                print(f"Viewing {file} (lines {start+1}-{min(start+page_size, len(lines))} of {len(lines)})")
                print("="*80)
                print('\n'.join(lines[start:start+page_size]))
                print("="*80)
                
                if start + page_size < len(lines):
                    nav = input("\nPress Enter for next page, 'q' to quit: ").strip().lower()
                    if nav == 'q':
                        break
                    start += page_size
                else:
                    input("\nEnd of file. Press Enter to continue...")
                    break
        else:
            print(f"\nFile {file} not found!")
            input("Press Enter to continue...")


def run_all():
    """Run all demonstrations in sequence."""
    clear_screen()
    print("RUNNING ALL DEMONSTRATIONS")
    print("="*80)
    print("\nThis will run:")
    print("1. Three step-by-step demonstrations")
    print("2. Comprehensive comparison")
    print("3. Edge cases analysis")
    print("4. Mathematical proofs")
    
    confirm = input("\nThis will take several minutes. Proceed? (y/n): ").strip().lower()
    if confirm == 'y':
        scripts = [
            'demonstrations/step1_linear_equivalence.py',
            'demonstrations/step2_nonlinear_dml.py',
            'demonstrations/step3_interactions_causal.py',
            'demonstrations/comprehensive_comparison.py',
            'demonstrations/edge_cases_demo.py',
            'notebooks/mathematical_proofs.py'
        ]
        
        for i, script in enumerate(scripts, 1):
            print(f"\n[{i}/{len(scripts)}] Running {script}...")
            subprocess.run([sys.executable, script])
            if i < len(scripts):
                input("\nPress Enter for next demonstration...")
        
        print("\nAll demonstrations complete!")
        print("\nCheck the outputs/ directory for results:")
        print("- outputs/figures/ - All visualizations")
        print("- outputs/tables/ - Numerical results")
        input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            run_step_demos()
        elif choice == '2':
            run_comprehensive()
        elif choice == '3':
            run_edge_cases()
        elif choice == '4':
            run_diagnostic()
        elif choice == '5':
            view_math()
        elif choice == '6':
            view_docs()
        elif choice == '7':
            run_all()
        elif choice == '0':
            print("\nThank you for using the Mediation Frameworks Comparison toolkit!")
            print("For questions or issues, please refer to the documentation.")
            break
        else:
            print("Invalid choice! Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('src') or not os.path.exists('demonstrations'):
        print("Error: Please run this script from the poma_frameworks_comparison directory")
        sys.exit(1)
    
    # Create output directories if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/tables', exist_ok=True)
    
    # Run main menu
    main()