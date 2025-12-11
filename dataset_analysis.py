from datasets import load_dataset
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = Path("plots")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [
        "president",
        "interview_question",
        "interview_answer",
        "question",
        "clarity_label",
        "evasion_label",
    ]
    existing_cols = [col for col in columns_to_keep if col in df.columns]
    return df[existing_cols]


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['question_length'] = df['question'].str.len()
    df['answer_length'] = df['interview_answer'].str.len()
    df['question_word_count'] = df['question'].str.split().str.len()
    df['answer_word_count'] = df['interview_answer'].str.split().str.len()
    return df


def plot_label_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    clarity_counts = df['clarity_label'].value_counts()
    axes[0].pie(clarity_counts, labels=clarity_counts.index, autopct='%1.1f%%')
    axes[0].set_title('Clarity Label Distribution', fontsize=14, fontweight='bold')
    
    evasion_counts = df['evasion_label'].value_counts()
    sns.barplot(x=evasion_counts.values, y=evasion_counts.index, hue=evasion_counts.index, 
                ax=axes[1], palette='viridis', legend=False)
    axes[1].set_xlabel('Count')
    axes[1].set_title('Evasion Label Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'label_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_text_length_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(df['question_length'], kde=True, ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Question Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Character Count')
    
    sns.histplot(df['answer_length'], kde=True, ax=axes[0, 1], color='purple')
    axes[0, 1].set_title('Answer Length Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Character Count')
    
    sns.boxplot(data=df, x='clarity_label', y='answer_length', hue='clarity_label',
                ax=axes[1, 0], palette='Set2', legend=False)
    axes[1, 0].set_title('Answer Length by Clarity Label', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=15)

    taxonomy_order = [
        'Explicit',           
        'Implicit',           
        'Dodging',
        'General',
        'Deflection',
        'Partial/half-answer',
        'Declining to answer', 
        'Claims ignorance',
        'Clarification',
    ]
    sns.boxplot(data=df, x='evasion_label', y='answer_length', hue='evasion_label',
                ax=axes[1, 1], order=taxonomy_order, palette='coolwarm', legend=False)
    axes[1, 1].set_title('Answer Length by Evasion Label', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'text_length_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_president_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pres_counts = df['president'].value_counts()
    sns.barplot(x=pres_counts.values, y=pres_counts.index, hue=pres_counts.index,
                ax=axes[0], palette='Blues_d', legend=False)
    axes[0].set_title('Samples per President', fontweight='bold')
    axes[0].set_xlabel('Count')
    axes[0].bar_label(axes[0].containers[0], padding=3)
    
    pres_clarity = pd.crosstab(df['president'], df['clarity_label'], normalize='index') * 100
    pres_clarity.plot(kind='barh', stacked=True, ax=axes[1], colormap='RdYlGn')
    axes[1].set_title('Clarity Distribution by President', fontweight='bold')
    axes[1].set_xlabel('Percentage (%)')
    axes[1].legend(title='Clarity', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'president_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_summary_statistics(df: pd.DataFrame):
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    numeric_cols = ['question_length', 'answer_length', 'question_word_count', 'answer_word_count']
    print(df[numeric_cols].describe().round(2))


if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)
    
    dataset = load_dataset("ailsntua/QEvasion")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    
    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print(f"Total columns: {len(df_train.columns)}")
    
    df = clean_dataset(df_train)
    print(f"\nKept columns: {list(df.columns)}")
    
    df = add_text_features(df)
    
    print_summary_statistics(df)
    
    print(f"\n{'='*60}")
    print("LABEL DISTRIBUTIONS")
    print("="*60)
    print("\nClarity Labels:")
    print(df['clarity_label'].value_counts())
    print("\nEvasion Labels:")
    print(df['evasion_label'].value_counts())
    
    
    plot_label_distribution(df)
    plot_text_length_analysis(df)
    plot_president_analysis(df)

