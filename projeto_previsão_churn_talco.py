import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score ,confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, fbeta_score
from imblearn.combine import SMOTETomek
from feature_engineering import apply_custom_features
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from plotagem import plot_histogram, plot_stacked_bar, plot_confusion_matrix, plot_roc_curve

# File Path
file_path = r'C:\Users\Luiz Gustavo\Desktop\Projeto Previsão de Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv'

# ==============================================
# DATA LOADING AND CLEANING (ORIGINAL)
# ==============================================

df = pd.read_csv(file_path)
df.drop('customerID', axis=1, inplace=True)

def convert_total_charges(series):
    """Convert TotalCharges to numeric, handling empty strings and spaces"""
    return pd.to_numeric(
        series.astype(str).str.strip().replace('', np.nan),
        errors='coerce'
    ).fillna(0)

df['TotalCharges'] = convert_total_charges(df['TotalCharges'])

for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

df.fillna({
    **{col: 'Unknown' for col in df.select_dtypes(include=['object']).columns},
    **{col: df[col].median() for col in df.select_dtypes(include=['number']).columns} 
}, inplace=True)

df.drop_duplicates(inplace=True)

print("\n[1/6] === NULL VALUES CHECK ===")
print("Total null values per column:")
print(df.isnull().sum())

print("\n[2/6] === DUPLICATES CHECK ===")
print("Total duplicate rows:", df.duplicated().sum())

print("\n[3/6] === DATA TYPES VERIFICATION ===")
print(df.dtypes)

print("\n[4/6] === CATEGORICAL VALUES CLEANING ===")
for col in df.select_dtypes(include='object').columns:
    empty_count = (df[col].astype(str).str.strip() == '').sum()
    print(f"Column '{col}': {empty_count} empty strings remaining")

print("\n[5/6] === NUMERIC RANGE VALIDATION ===")
print("Tenure range:", (df['tenure'].min(), df['tenure'].max()))
print("TotalCharges range:", (df['TotalCharges'].min(), df['TotalCharges'].max()))

print("\n[6/6] === SAMPLE DATA CHECK ===")
print("Random 5 rows sample:")
print(df.sample(5, random_state=1))
print("\nShape do DataFrame:", df.shape)

# ==============================================
# FEATURE ENGINEERING (ORIGINAL + NEW)
# ==============================================

def create_features(X):
    """Enhanced feature engineering with robust handling"""
    X = X.copy()
    # tenure group
    X['tenure_group'] = pd.cut(
        X['tenure'],
        bins=[0,6,12,24,60,72],
        labels=['0-6','7-12','13-24','25-60','61-72']
    )
    # avg per tenure
    X['avg_charge_per_tenure'] = X['TotalCharges'] / np.where(X['tenure'] == 0, 1, X['tenure'])
    # high value
    X['high_value_flag'] = (
        (X['MonthlyCharges'] > X['MonthlyCharges'].quantile(0.75)) & 
        (X['tenure'] > X['tenure'].median())
    ).astype(int)
    # service density
    service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    existing = [c for c in service_cols if c in X.columns]
    if existing:
        X['service_density'] = X[existing].apply(
            lambda row: row.str.contains('Yes').sum(), axis=1
        )
    else:
        X['service_density'] = 0
    return X

# ==============================================
# MACHINE LEARNING PIPELINE
# ==============================================

X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes':1,'No':0})

X = create_features(X)


numeric_features = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen', 'avg_charge_per_tenure', 'service_density', 'high_value_flag']
categorical_features = [c for c in X.columns if c not in numeric_features]


preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', Pipeline([ 
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler()) 
        ]), numeric_features), 
        ('cat', Pipeline([ 
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
            ('encoder', OneHotEncoder(handle_unknown='ignore')) 
        ]), categorical_features) 
    ]
)

pipeline = Pipeline([ 
    ('create_features', FunctionTransformer(create_features, validate=False)), 
    ('preprocessor', preprocessor), 
    # oversample 20% 
    ('smote', SMOTETomek(sampling_strategy=1.0, random_state=42)), 
    ('classifier', RandomForestClassifier( 
        class_weight={0:1, 1:5},  # more weight for churn 
        random_state=42 
    )) 
])

# ==============================================
# HYPERPARAMETER TUNING (focus on F2)
# ==============================================

param_dist = {
    'classifier__n_estimators': [100, 200, 300, 400],
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', 0.5],
    'classifier__class_weight': [None, 'balanced', {0:1, 1:3}, {0:1, 1:5}],
    'smote__sampling_strategy': [0.5, 0.7, 0.8, 1.0]
}


# F2 Scorer for RandomizedSearchCV
f2_scorer = make_scorer(fbeta_score, beta=2)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=40, cv=cv,
    scoring=f2_scorer, verbose=1, n_jobs=-1, random_state=42
)

# ==============================================
# Calling Feature Engineering to increase precision

df = apply_custom_features(df)
# ==============================================

# ==============================================
# ==============================================
# TRAIN-TEST SPLIT & MODEL FIT
# ==============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n[Data Validation] Missing X_train:", X_train.isnull().sum().sum())
print("Class dist:", y_train.value_counts(normalize=True))

search.fit(X_train, y_train)
best_model = search.best_estimator_

# ==============================================
# THRESHOLD EXPLORATION
# ==============================================

y_probs = best_model.predict_proba(X_test)[:, 1]
print("\nThreshold vs recall/precision/F1 for churn:")
for t in np.linspace(0.3, 0.9, 7):
    preds = (y_probs >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True)
    print(f"t={t:.2f}: recall1={rep['1']['recall']:.2f}, prec1={rep['1']['precision']:.2f}, f1={rep['1']['f1-score']:.2f}")

# Optimizing threshold using F2
prec, rec, thr = precision_recall_curve(y_test, y_probs)
f2_scores = (1 + 2**2) * (prec * rec) / (4 * prec + rec)
best_idx = np.nanargmax(f2_scores)
best_t = thr[best_idx]
print(f"\nBest threshold for F2: {best_t:.2f}")

# ==============================================
# FINAL EVALUATION
# ==============================================

y_pred = (y_probs >= best_t).astype(int)
print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Optimal Threshold: {best_t:.2f}")


def print_metrics(y_true, y_pred, y_probs):
    """Imprime métricas de classificação no terminal"""
    
    # Métricas básicas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Formatação
    print("\n" + "="*40)
    print("=== Métricas de Desempenho ===")
    print("="*40)
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão (Churn): {prec:.4f}")
    print(f"Recall (Churn): {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    print("\n=== Matriz de Confusão ===")
    print(f"                Previsto 0   Previsto 1")
    print(f"Real 0 (Não Churn)   {tn:8}      {fp:8}")
    print(f"Real 1 (Churn)       {fn:8}      {tp:8}")

print("ROC AUC:", roc_auc_score(y_test, y_probs))
print("Average Precision:", average_precision_score(y_test, y_probs))
# ==============================================
#  PROPOSED IMPROVEMENTS 
# ==============================================

# Automatic optimal threshold calculation
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls)
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n[INFO] Optimal threshold based on F2-score: {optimal_threshold:.2f}")

# Dynamic churn risk classification
def classify_risk(score):
    """
    Classify customer churn risk based on probability score and optimal threshold.

    Parameters:
    - score (float): Churn probability score.

    Returns:
    - str: Risk category ('High', 'Moderate', or 'Low').
    """
    if score >= optimal_threshold + 0.25:
        return 'High'
    elif score >= optimal_threshold:
        return 'Moderate'
    else:
        return 'Low'

# Create result DataFrame
df_result = X_test.copy()
df_result['y_true'] = y_test.values
df_result['y_prob'] = y_probs
df_result['y_pred'] = y_pred
df_result['Prediction_Type'] = df_result.apply(
    lambda row: 'TP' if row['y_true'] == 1 and row['y_pred'] == 1 else
                'FP' if row['y_true'] == 0 and row['y_pred'] == 1 else
                'FN' if row['y_true'] == 1 and row['y_pred'] == 0 else
                'TN',
    axis=1
)

df_result['Risk_Level'] = df_result['y_prob'].apply(classify_risk)

# Cost calculation based on prediction errors
custo_fp = 100   # Retention campaign cost for a false positive
custo_fn = 2000  # Estimated lost revenue for a false negative

df_result['Estimated_Cost'] = np.select(
    [
        df_result['Prediction_Type'] == 'FP',
        df_result['Prediction_Type'] == 'FN'
    ],
    [custo_fp, custo_fn],
    default=0
)

# Exporting to CSV for further analysis
df_result.to_csv(
    'clientes_classificados.csv',
    index=False,
    sep=';',           
    encoding='utf-8-sig',
    float_format='%.2f',
    decimal=','   

)
print("Saved clientes_classificados.csv with UTF-8 BOM.")

# Show full DataFrame in terminal
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print("\n[INFO] Classified customers with risk levels and costs:")
#print(df_result)


# Precision-Recall Curve
#plt.figure(figsize=(8,6))
#plt.plot(rec, prec, label='PR Curve')
#plt.axvline(best_t, color='red', linestyle='--', label=f'T={best_t:.2f}')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Trade-off')
#plt.legend()
#plt.show()

#===============================
# PLOTAGEM
#==============================
# Page configuration
st.set_page_config(layout="wide", page_title="Churn Dashboard")

# Load data (example)
# Substitua a função load_data() por:

@st.cache_data
def load_data():
    """Carrega e processa os dados reais"""
    file_path = r'C:\Users\Luiz Gustavo\Desktop\Projeto Previsão de Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Carrega os dados
    df = pd.read_csv(file_path)
    df.drop('customerID', axis=1, inplace=True)
    
    # Processamento dos dados (igual ao que você já tem)
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'].astype(str).str.strip().replace('', np.nan),
        errors='coerce'
    ).fillna(0)
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
    
    df.fillna({
        **{col: 'Unknown' for col in df.select_dtypes(include=['object']).columns},
        **{col: df[col].median() for col in df.select_dtypes(include=['number']).columns} 
    }, inplace=True)
    
    df.drop_duplicates(inplace=True)
    
    # Aplica feature engineering
    df = apply_custom_features(df)
    
    # Converte Churn para numérico
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df
# Carrega os dados REAIS
df = load_data()

# --- Sidebar ---
with st.sidebar:
    st.header("Filters")
    contract_filter = st.multiselect(
        "Contract Type",
        options=df['Contract'].unique(),
        default=df['Contract'].unique()
    )

# Aplica filtros
filtered_df = df[df['Contract'].isin(contract_filter)]

# --- Section 1: Charts ---
st.header("Churn Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Tenure Distribution")
    fig1 = plot_histogram(filtered_df, 'tenure', 'Customer Tenure')
    st.pyplot(fig1)

with col2:
    st.subheader("Churn by Contract Type")
    
    # Verificação dos dados
    st.write("Distribuição real:", 
             filtered_df.groupby('Contract')['Churn'].mean().round(2))
    
    # Geração do gráfico
    fig2 = plot_stacked_bar(filtered_df, 'Contract', 'Churn', 'Churn by Contract Type')
    if fig2:
        fig2.update_layout(
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
# --- NOVA SEÇÃO DE INSIGHTS ---
st.header("🔍 Análise de Insights")

col_insight1, col_insight2 = st.columns(2)
with col_insight1:
    st.markdown("""
    **📊 Churn por Tipo de Contrato:**
    - Clientes mensais têm risco **14x maior** que clientes com contrato de 2 anos
    - 43% dos clientes mensais deixam o serviço vs apenas 3% dos clientes anuais
    """)

with col_insight2:
    st.markdown("""
    **🎯 Recomendações Ações:**
    - Oferecer desconto progressivo para conversão em contratos anuais
    - Criar programa de fidelidade para clientes mensais
    """)

# --- Section 2: Model Validation ---
# [Seu conteúdo atual de validação...]

# --- Section 2: Model Validation ---
st.header("Model Validation")

# Synthetic example data
y_true_real = df_result['y_true'].values
y_probs_real = df_result['y_prob'].values
y_pred_real = df_result['y_pred'].values

col3, col4 = st.columns(2)
with col3:
    st.subheader("Confusion Matrix")
    fig3 = plot_confusion_matrix(y_true_real, y_pred_real, "Confusion Matrix")
    st.pyplot(fig3)

with col4:
    st.subheader("ROC Curve")
    fig4 = plot_roc_curve(y_true_real, y_probs_real, "ROC Curve")
    st.pyplot(fig4)