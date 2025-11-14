import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os


COMBINED_PATH = '../../metrics/combined_evaluation_results.csv'
ENSEMBLE_TYPE_TO_PLOT = 'ROBERTA_Ensemble' 
BASE_MODEL_FAMILY_TO_COMPARE = 'ROBERTA'

try:
    combined_df = pd.read_csv(COMBINED_PATH)
except FileNotFoundError:
    print("exiting")
    sys.exit(1)

base_df = combined_df[combined_df['improvement'].isna()].copy()
ensemble_summary_df_raw = combined_df[combined_df['improvement'].notna()].copy()

base_df['model'] = base_df['model'].fillna('UNKNOWN') 
base_df['model'] = base_df['model'].replace({
    'Baseline': 'TF-IDF + LogReg',
    'Transformer': 'RoBERTa/BERT/DistilBERT'
})
base_df['train_domain'] = base_df['train_domain'].fillna('UNKNOWN') 

if not ensemble_summary_df_raw.empty:
    num_ensembles_per_type = 3 
    ensemble_types_sequence = ['BERT_Ensemble', 'DISTILBERT_Ensemble', 'ROBERTA_Ensemble'] 
    
    if len(ensemble_summary_df_raw) == len(ensemble_types_sequence) * num_ensembles_per_type:
        ensemble_summary_df_raw['ensemble_type'] = '' 
        for i, ensemble_type in enumerate(ensemble_types_sequence):
            start_idx = i * num_ensembles_per_type
            end_idx = (i + 1) * num_ensembles_per_type
            ensemble_summary_df_raw.loc[
                ensemble_summary_df_raw.index[start_idx:end_idx], 'ensemble_type'
            ] = ensemble_type
    else:
        ensemble_summary_df_raw['ensemble_type'] = 'Generic_Ensemble'

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.titlesize'] = 14 
plt.rcParams['xtick.labelsize'] = 11 
plt.rcParams['ytick.labelsize'] = 11 
plt.rcParams['legend.fontsize'] = 10 

plt.figure(figsize=(10, 6))
model_avg_f1 = base_df.groupby('model')['f1'].mean().sort_values(ascending=False).reset_index()
model_avg_f1.columns = ['Model', 'Average F1 Score']
sns.barplot(
    data=model_avg_f1,
    x='Model',
    y='Average F1 Score',
    hue='Model',
    palette='Spectral',
    legend=False
)
plt.title('1. Overall Average F1 Score by Model Family', fontsize=14, fontweight='bold')
plt.xlabel('Model Family', fontsize=12)
plt.ylabel('Average F1 Score (Mean of 9 Scenarios)', fontsize=12)
plt.ylim(base_df['f1'].min() - 0.01, 1.0)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('figures/viz_1_overall_model_average_f1.png', dpi=300)

plt.figure(figsize=(12, 7))
base_df_for_bert = combined_df[combined_df['improvement'].isna()].copy()
df_transfer = base_df_for_bert[base_df_for_bert['model'] == 'BERT'].copy()
sns.barplot(
    data=df_transfer,
    x='test_domain',
    y='f1',
    hue='train_domain',
    palette='Set1'
)
plt.title('2. Cross-Domain Transfer Challenge (BERT Family F1 Score)', fontsize=14, fontweight='bold')
plt.xlabel('Test Domain', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.ylim(base_df['f1'].min() - 0.02, 1.0)
plt.legend(title='Training Domain', loc='upper right')
plt.tight_layout()
plt.savefig('figures/viz_2_cross_domain_impact.png', dpi=300)

if not ensemble_summary_df_raw.empty:
    plt.figure(figsize=(10, 6))
    best_base_f1_overall = base_df.groupby('test_domain')['f1'].max().reset_index()
    best_base_f1_overall.rename(columns={'f1': 'Best Base Model F1 (Overall)'}, inplace=True)
    plot_df_3 = pd.merge(ensemble_summary_df_raw, best_base_f1_overall, on='test_domain')
    df_ensemble_plot_3 = plot_df_3.melt(
        id_vars=['test_domain', 'ensemble_type'],
        value_vars=['ensemble_f1', 'Best Base Model F1 (Overall)'],
        var_name='Metric Type',
        value_name='F1 Score'
    )
    df_ensemble_plot_3['Metric Type'] = df_ensemble_plot_3['Metric Type'].replace({
        'ensemble_f1': 'Ensemble F1 Score',
        'Best Base Model F1 (Overall)': 'Best Base Model F1 (Overall)'
    })
    df_ensemble_plot_3_filtered = df_ensemble_plot_3[
        df_ensemble_plot_3['ensemble_type'] == 'ROBERTA_Ensemble' 
    ].copy()
    sns.barplot(
        data=df_ensemble_plot_3_filtered,
        x='test_domain',
        y='F1 Score',
        hue='Metric Type',
        palette=['#1f77b4', '#ff7f0e']
    )
    plt.title('3. Ensemble Performance vs. Best Base Model F1 (Overall)', fontsize=14, fontweight='bold')
    plt.xlabel('Test Domain', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0.85, 1.0)
    plt.legend(title='Metric Source')
    plt.tight_layout()
    plt.savefig('figures/viz_3_ensemble_comparison_generic.png', dpi=300)

if not ensemble_summary_df_raw.empty:
    df_ensemble_plot_selected_4 = ensemble_summary_df_raw[
        ensemble_summary_df_raw['ensemble_type'] == ENSEMBLE_TYPE_TO_PLOT
    ].copy()
    if not df_ensemble_plot_selected_4.empty:
        plt.figure(figsize=(14, 8)) 
        filtered_base_df_4 = combined_df[
            (combined_df['improvement'].isna()) & 
            (combined_df['model'] == BASE_MODEL_FAMILY_TO_COMPARE)
        ].copy()
        filtered_base_df_4['Model_TrainDomain'] = filtered_base_df_4['model'] + ' (' + filtered_base_df_4['train_domain'] + ')'
        all_roberta_models_f1_for_plot_4 = filtered_base_df_4[['test_domain', 'f1', 'Model_TrainDomain']].copy()
        all_roberta_models_f1_for_plot_4.rename(columns={'f1': 'F1 Score', 'Model_TrainDomain': 'Model Type'}, inplace=True)
        ensemble_f1_data_4 = df_ensemble_plot_selected_4[['test_domain', 'ensemble_f1']].copy()
        ensemble_f1_data_4.rename(columns={'ensemble_f1': 'F1 Score'}, inplace=True)
        ensemble_f1_data_4['Model Type'] = f'{ENSEMBLE_TYPE_TO_PLOT} (Our Ensemble)' 
        plot_df_bars_4 = pd.concat([all_roberta_models_f1_for_plot_4, ensemble_f1_data_4], ignore_index=True)
        ensemble_model_type_name_4 = f'{ENSEMBLE_TYPE_TO_PLOT} (Our Ensemble)'
        plot_df_bars_4['Sort_Key'] = plot_df_bars_4['Model Type'].apply(
            lambda x: 0 if x == ensemble_model_type_name_4 else 1
        )
        plot_df_bars_4 = plot_df_bars_4.sort_values(by=['test_domain', 'Sort_Key', 'Model Type'], ascending=[True, True, True])
        plot_df_bars_4.drop(columns='Sort_Key', inplace=True)
        unique_model_types_4 = plot_df_bars_4['Model Type'].unique()
        other_model_types_4 = [mt for mt in unique_model_types_4 if mt != ensemble_model_type_name_4]
        base_colors_4 = sns.color_palette('Blues', n_colors=len(other_model_types_4))
        custom_palette_4 = {ensemble_model_type_name_4: 'darkorange'} 
        for i, model_type in enumerate(other_model_types_4):
            custom_palette_4[model_type] = base_colors_4[i]
        ax4 = sns.barplot(
            data=plot_df_bars_4,
            x='test_domain',
            y='F1 Score',
            hue='Model Type', 
            palette=custom_palette_4, 
            errorbar=None,
        )
        plt.xticks(rotation=0, ha='center') 
        ax4.legend(title='Model Type', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title_fontsize=12)
        plt.title(f'4. {ENSEMBLE_TYPE_TO_PLOT} vs. All {BASE_MODEL_FAMILY_TO_COMPARE} Base Models', fontsize=16, fontweight='bold')
        plt.xlabel('Test Domain', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.ylim(plot_df_bars_4['F1 Score'].min() * 0.95, 1.0) 
        plt.tight_layout(rect=[0, 0, 0.75, 1]) 
        plt.savefig(f'figures/viz_4_{BASE_MODEL_FAMILY_TO_COMPARE.lower()}_ensemble_comparison.png', dpi=300)