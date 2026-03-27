#!/usr/bin/env python3
"""
Create Architecture Diagram - Generate visual representation of multi-label strategies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_chained_diagram():
    """Create diagram for Chained Multi-Outputs Strategy"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Design Decision 1: Chained Multi-Outputs Strategy', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input Data
    input_box = FancyBboxPatch((0.5, 8), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.4, 'Input Data\n(X)', fontsize=12, ha='center', va='center')
    
    # Model 1: Type 2
    model1_box = FancyBboxPatch((3.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(model1_box)
    ax.text(4.5, 8.4, 'Model 1\nType 2', fontsize=12, ha='center', va='center')
    
    # Arrow 1
    ax.arrow(2.5, 8.4, 1.0, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black')
    
    # Prediction 1
    pred1_box = FancyBboxPatch((6.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(pred1_box)
    ax.text(7.5, 8.4, 'Pred 1\nType 2', fontsize=12, ha='center', va='center')
    
    # Arrow 2
    ax.arrow(5.5, 8.4, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Combine for Model 2
    combine_box = FancyBboxPatch((0.5, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(combine_box)
    ax.text(1.5, 6.9, 'X + Pred 1\nCombined', fontsize=12, ha='center', va='center')
    
    # Arrow 3
    ax.arrow(2.5, 8.4, 0, -1.2, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    ax.arrow(7.5, 8.4, -4.5, -1.2, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Model 2: Type 2+3
    model2_box = FancyBboxPatch((3.5, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(model2_box)
    ax.text(4.5, 6.9, 'Model 2\nType 2+3', fontsize=12, ha='center', va='center')
    
    # Arrow 4
    ax.arrow(2.5, 6.9, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Prediction 2
    pred2_box = FancyBboxPatch((6.5, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(pred2_box)
    ax.text(7.5, 6.9, 'Pred 2\nType 2+3', fontsize=12, ha='center', va='center')
    
    # Arrow 5
    ax.arrow(5.5, 6.9, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Combine for Model 3
    combine2_box = FancyBboxPatch((0.5, 5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(combine2_box)
    ax.text(1.5, 5.4, 'X + Pred 1 + Pred 2\nCombined', fontsize=12, ha='center', va='center')
    
    # Arrow 6
    ax.arrow(2.5, 6.9, 0, -1.2, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    ax.arrow(7.5, 6.9, -4.5, -1.2, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Model 3: Type 2+3+4
    model3_box = FancyBboxPatch((3.5, 5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(model3_box)
    ax.text(4.5, 5.4, 'Model 3\nType 2+3+4', fontsize=12, ha='center', va='center')
    
    # Arrow 7
    ax.arrow(2.5, 5.4, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Final Prediction
    final_box = FancyBboxPatch((6.5, 5), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='gold', edgecolor='darkgoldenrod', linewidth=2)
    ax.add_patch(final_box)
    ax.text(7.5, 5.4, 'Final\nPrediction', fontsize=12, ha='center', va='center')
    
    # Arrow 8
    ax.arrow(5.5, 5.4, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Chain Accuracy
    chain_box = FancyBboxPatch((3.5, 3.5), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lavender', edgecolor='purple', linewidth=2)
    ax.add_patch(chain_box)
    ax.text(4.5, 3.9, 'Chain Accuracy\nMetric', fontsize=12, ha='center', va='center')
    
    # Arrow 9
    ax.arrow(4.5, 5.0, 0, -0.7, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Characteristics
    ax.text(1, 2.5, 'Characteristics:', fontsize=12, fontweight='bold')
    ax.text(1, 2.2, '• Sequential dependency', fontsize=10)
    ax.text(1, 2.0, '• Each level depends on previous accuracy', fontsize=10)
    ax.text(1, 1.8, '• Chain accuracy evaluation', fontsize=10)
    ax.text(1, 1.6, '• Results: 100% accuracy', fontsize=10)
    
    ax.text(6, 2.5, 'Use Cases:', fontsize=12, fontweight='bold')
    ax.text(6, 2.2, '• When sequential accuracy is critical', fontsize=10)
    ax.text(6, 2.0, '• When prediction order matters', fontsize=10)
    ax.text(6, 1.8, '• When chain dependencies exist', fontsize=10)
    ax.text(6, 1.6, '• When intermediate predictions are valuable', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/chained_strategy_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hierarchical_diagram():
    """Create diagram for Hierarchical Multi-Label Strategy"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Design Decision 2: Hierarchical Multi-Label Strategy', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input Data
    input_box = FancyBboxPatch((0.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='navy', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.4, 'Input Data\n(X)', fontsize=12, ha='center', va='center')
    
    # Data Filter 1
    filter1_box = FancyBboxPatch((3.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(filter1_box)
    ax.text(4.5, 8.4, 'Data Filter\nAll samples', fontsize=12, ha='center', va='center')
    
    # Arrow 1
    ax.arrow(2.5, 8.4, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Model 1: Type 2
    model1_box = FancyBboxPatch((6.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(model1_box)
    ax.text(7.5, 8.4, 'Model 1\nType 2', fontsize=12, ha='center', va='center')
    
    # Arrow 2
    ax.arrow(5.5, 8.4, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Prediction 1
    pred1_box = FancyBboxPatch((6.5, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(pred1_box)
    ax.text(7.5, 6.9, 'Prediction 1\nType 2', fontsize=12, ha='center', va='center')
    
    # Arrow 3
    ax.arrow(7.5, 8.0, 0, -0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Data Filter 2
    filter2_box = FancyBboxPatch((3.5, 6.5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(filter2_box)
    ax.text(4.5, 6.9, 'Data Filter\nFiltered by Type 2', fontsize=12, ha='center', va='center')
    
    # Arrow 4
    ax.arrow(2.5, 8.4, 0, -1.2, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Model 2: Type 3 (Multiple)
    model2_box = FancyBboxPatch((6.5, 5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(model2_box)
    ax.text(7.5, 5.4, 'Model 2\nType 3 (N models)', fontsize=12, ha='center', va='center')
    
    # Arrow 5
    ax.arrow(5.5, 6.9, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Prediction 2
    pred2_box = FancyBboxPatch((6.5, 3.5), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(pred2_box)
    ax.text(7.5, 3.9, 'Prediction 2\nType 3', fontsize=12, ha='center', va='center')
    
    # Arrow 6
    ax.arrow(7.5, 5.0, 0, -0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Data Filter 3
    filter3_box = FancyBboxPatch((3.5, 5), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(filter3_box)
    ax.text(4.5, 5.4, 'Data Filter\nFiltered by Type 2+3', fontsize=12, ha='center', va='center')
    
    # Arrow 7
    ax.arrow(2.5, 6.9, 0, -1.2, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Model 3: Type 4 (Multiple)
    model3_box = FancyBboxPatch((6.5, 2), 2, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(model3_box)
    ax.text(7.5, 2.4, 'Model 3\nType 4 (M models)', fontsize=12, ha='center', va='center')
    
    # Arrow 8
    ax.arrow(5.5, 5.4, 1.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Final Prediction
    final_box = FancyBboxPatch((6.5, 0.5), 2, 0.8, boxstyle="round,pad=0.1",
                               facecolor='gold', edgecolor='darkgoldenrod', linewidth=2)
    ax.add_patch(final_box)
    ax.text(7.5, 0.9, 'Final\nPrediction', fontsize=12, ha='center', va='center')
    
    # Arrow 9
    ax.arrow(7.5, 2.0, 0, -0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Hierarchical Accuracy
    hier_box = FancyBboxPatch((1, 3), 2, 0.8, boxstyle="round,pad=0.1",
                              facecolor='lavender', edgecolor='purple', linewidth=2)
    ax.add_patch(hier_box)
    ax.text(2, 3.4, 'Hierarchical\nAccuracy\nMetric', fontsize=12, ha='center', va='center')
    
    # Arrow 10
    ax.arrow(3.0, 3.4, 3.0, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Characteristics
    ax.text(1, 2.0, 'Characteristics:', fontsize=12, fontweight='bold')
    ax.text(1, 1.7, '• Multiple specialized models', fontsize=10)
    ax.text(1, 1.5, '• Data filtering at each level', fontsize=10)
    ax.text(1, 1.3, '• Independent model training', fontsize=10)
    ax.text(1, 1.1, '• Results: 100% accuracy', fontsize=10)
    
    ax.text(6, 2.0, 'Use Cases:', fontsize=12, fontweight='bold')
    ax.text(6, 1.7, '• When class-specific models are preferred', fontsize=10)
    ax.text(6, 1.5, '• When data filtering is beneficial', fontsize=10)
    ax.text(6, 1.3, '• When independent training is needed', fontsize=10)
    ax.text(6, 1.1, '• When specialized models perform better', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/hierarchical_strategy_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_diagram():
    """Create comparison diagram between both strategies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chained Strategy
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.text(5, 9.5, 'Chained Multi-Outputs', fontsize=14, fontweight='bold', ha='center')
    
    # Simplified chained flow
    boxes_chained = [
        (1, 8.5, 'Input'),
        (3, 8.5, 'Model 1'),
        (5, 8.5, 'Model 2'),
        (7, 8.5, 'Model 3'),
        (9, 8.5, 'Output')
    ]
    
    for i, (x, y, label) in enumerate(boxes_chained):
        box = FancyBboxPatch((x-0.5, y-0.4), 1, 0.8, boxstyle="round,pad=0.05",
                            facecolor='lightblue', edgecolor='navy', linewidth=1)
        ax1.add_patch(box)
        ax1.text(x, y, label, fontsize=10, ha='center', va='center')
        
        if i < len(boxes_chained) - 1:
            ax1.arrow(x+0.5, y, 0.8, 0, head_width=0.1, head_length=0.1,
                     fc='black', ec='black')
    
    ax1.text(5, 7, 'Sequential Chain', fontsize=12, ha='center', style='italic')
    ax1.text(5, 6.5, '• 3 models in sequence', fontsize=10, ha='center')
    ax1.text(5, 6, '• Chain dependency', fontsize=10, ha='center')
    ax1.text(5, 5.5, '• 100% accuracy', fontsize=10, ha='center')
    
    # Hierarchical Strategy
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.text(5, 9.5, 'Hierarchical Multi-Label', fontsize=14, fontweight='bold', ha='center')
    
    # Simplified hierarchical flow
    # Model 1
    box1 = FancyBboxPatch((1, 8), 2, 0.8, boxstyle="round,pad=0.05",
                         facecolor='lightgreen', edgecolor='darkgreen', linewidth=1)
    ax2.add_patch(box1)
    ax2.text(2, 8.4, 'Model 1', fontsize=10, ha='center', va='center')
    
    # Models 2 and 3
    box2 = FancyBboxPatch((4, 8), 2, 0.8, boxstyle="round,pad=0.05",
                         facecolor='lightgreen', edgecolor='darkgreen', linewidth=1)
    ax2.add_patch(box2)
    ax2.text(5, 8.4, 'Model 2 (N)', fontsize=10, ha='center', va='center')
    
    box3 = FancyBboxPatch((7, 8), 2, 0.8, boxstyle="round,pad=0.05",
                         facecolor='lightgreen', edgecolor='darkgreen', linewidth=1)
    ax2.add_patch(box3)
    ax2.text(8, 8.4, 'Model 3 (M)', fontsize=10, ha='center', va='center')
    
    # Arrows from input
    ax2.arrow(2, 8.0, 0, -0.5, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    ax2.arrow(5, 8.0, 0, -0.5, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    ax2.arrow(8, 8.0, 0, -0.5, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    # Combined output
    output_box = FancyBboxPatch((4, 6), 2, 0.8, boxstyle="round,pad=0.05",
                               facecolor='gold', edgecolor='darkgoldenrod', linewidth=1)
    ax2.add_patch(output_box)
    ax2.text(5, 6.4, 'Combined', fontsize=10, ha='center', va='center')
    
    # Arrows to output
    ax2.arrow(2, 7.5, 2.5, -0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    ax2.arrow(5, 7.5, 0, -0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    ax2.arrow(8, 7.5, -2.5, -0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black')
    
    ax2.text(5, 5, 'Parallel Processing', fontsize=12, ha='center', style='italic')
    ax2.text(5, 4.5, '• Multiple specialized models', fontsize=10, ha='center')
    ax2.text(5, 4, '• Data filtering', fontsize=10, ha='center')
    ax2.text(5, 3.5, '• 100% accuracy', fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all diagrams"""
    print("Creating architecture diagrams...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create diagrams
    create_chained_diagram()
    print("SUCCESS: Chained strategy diagram created")
    
    create_hierarchical_diagram()
    print("SUCCESS: Hierarchical strategy diagram created")
    
    create_comparison_diagram()
    print("SUCCESS: Strategy comparison diagram created")
    
    print("\nAll diagrams saved to 'results/' directory:")
    print("- chained_strategy_diagram.png")
    print("- hierarchical_strategy_diagram.png") 
    print("- strategy_comparison_diagram.png")

if __name__ == "__main__":
    main()
