import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score
import numpy as np
import streamlit as st

def plot_histogram(df, column, title, bins=None):
    """Plota histograma com KDE e formatação."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if bins is None:
        bins = 'auto'  # Flexível
    sns.histplot(df[column], kde=True, ax=ax, bins=bins)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Tempo de Permanência (anos)", fontsize=12)
    ax.set_ylabel("Contagem de Clientes", fontsize=12)
    if bins != 'auto':
        ax.set_xticks(bins[1:-1])
    return fig

def plot_stacked_bar(df, group_col, target_col, title):
    if df.empty:
        st.warning("O DataFrame está vazio. Não é possível gerar o gráfico.")
        return None

    try:
        # Verifica se as colunas existem
        if group_col not in df.columns or target_col not in df.columns:
            st.error(f"Colunas '{group_col}' ou '{target_col}' não encontradas no DataFrame.")
            return None

        # Agrupamento e cálculo de percentual
        df_counts = df.groupby([group_col, target_col]).size().unstack(fill_value=0)
        if df_counts.empty:
            st.warning("O agrupamento retornou um DataFrame vazio.")
            return None

        df_perc = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
        df_perc = df_perc.reset_index()

        fig = px.bar(
            df_perc,
            x=group_col,
            y=df_perc.columns[1:],  # ignora a primeira coluna (agrupador)
            title=title,
            labels={'value': 'Percentual', 'variable': target_col},
            barmode='stack'
        )
        fig.update_layout(yaxis_title='Percentual (%)')
        return fig

    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, title):
    """Matriz de confusão normalizada e com rótulos."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizar

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_normalized * 100 , 
        annot=True, 
        fmt='.2f',  # Mostrar porcentagens
        cmap='Blues', 
        ax=ax,
        xticklabels=['Não Churn', 'Churn'],  # Rótulos específicos
        yticklabels=['Não Churn', 'Churn']
    )
    ax.set_title(title, pad=20)
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    return fig


def plot_roc_curve(y_true, y_scores, title):
    """Curva ROC com cálculo dinâmico de AUC."""
    # Verificação de input
    if set(np.unique(y_scores)) <= {0, 1}:
        print("⚠️ Atenção: 'y_scores' parece conter apenas 0s e 1s. Recomenda-se passar as probabilidades (scores) para a curva ROC.")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='#FF6B6B')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    return fig
