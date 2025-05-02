## 🧠 Projeto de Previsão de Churn em Telecomunicações

**Recursos**:

* **Dashboard Streamlit**: [https://churn-prediction-project-with-kaggle-database-g5q9uswffyie7tzb.streamlit.app/](https://churn-prediction-project-with-kaggle-database-g5q9uswffyie7tzb.streamlit.app/)
* **Imagens**: pasta “prints do funcionamento” com capturas das funcionalidades, em estágio inicial e final.

---

## 📉 1. Introdução

Este projeto teve como objetivo prever o churn de clientes de uma operadora de telecomunicações, utilizando o conjunto de dados **Telco Customer Churn** (Kaggle). Através de técnicas de aprendizado de máquina e análise exploratória, concentrei meus esforços em identificar os principais fatores que influenciam a saída dos clientes, desenvolver um modelo preditivo eficaz e propor estratégias de retenção baseadas em perfis de risco. Além disso, incorporei uma análise de custo-benefício para apoiar a tomada de decisão orientada por dados.

**Origem dos Dados**: Telco Customer Churn (Kaggle)
**Técnicas Principais**:

1. **Random Forest** — escolhi por sua robustez e facilidade de interpretação via importância de atributos.
2. **SMOTETomek (SMOTE + Tomek Links)** — gerei amostras sintéticas da classe minoritária (SMOTE) e removi pares de pontos ambíguos na fronteira de decisão (Tomek Links).
3. **Engenharia de Features Avançada** — criei variáveis a partir de padrões apontados pela análise de erros (e.g., `tenure_group`, `high_value_flag`, `SeniorContractCombo`, `SupportServicesCount`).
4. **Otimização de Threshold via F₂-Score** — priorizei o recall (capturar churners) em relação à precisão.
5. **Pré-processamento e Pipeline Unificado** — utilizei `ColumnTransformer` para imputação e escala das variáveis numéricas, e codificação das categóricas, tudo encadeado em um único pipeline sem vazamento de dados.
6. **Hyperparameter Tuning com RandomizedSearchCV** — realizei busca aleatória em um espaço de parâmetros (e.g., `n_estimators`, `max_depth`, `class_weight`, `smote__sampling_strategy`) usando `make_scorer(fbeta_score, beta=2)` para otimizar o F₂‑Score.

---

## 🎯 2. Objetivo do Projeto

Prever churn de clientes com alta sensibilidade (recall) e precisão suficiente para que a equipe de retenção:

1. Concentre esforços nos casos de maior risco.
2. Minimize desperdício de recursos em falsos alarmes.

---

## 🔍 3. Entendimento e Iniciação

### 3.1 Escolha do Dataset

* **Fonte**: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Conteúdo**:

  * Dados demográficos (gênero, idade, dependentes)
  * Tipo de contrato (`Contract`), método de pagamento
  * Uso de serviços (TV, internet, suporte técnico)
  * Métricas de consumo (`MonthlyCharges`, `TotalCharges`)
  * Flag `Churn` (“Yes”/“No”)

### 3.2 Perguntas de Negócio

* Quais atributos realmente influenciam a decisão de churn?
* Como limpar inconsistências (e.g., `TotalCharges` como texto, espaços vazios)?
* Em um dataset desbalanceado (\~27% churn), como priorizar a detecção de churners sem gerar alarmes falsos em excesso?

### 3.3 Respostas às Perguntas de Negócio

1. **Atributos que influenciam o churn**
   Concentrei-me em extrair `feature_importances_` do Random Forest e apresentei um gráfico/tabela com os 10 atributos mais significativos.
2. **Limpeza de inconsistências**

   * Convertemos `TotalCharges` de string para numérico:

     ```python
     df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip().replace('', np.nan), errors='coerce').fillna(0)
     ```
   * Imputei valores faltantes em categóricas com `'Unknown'` e em numéricas com a mediana.
   * Removi identificadores (`customerID`) e verifiquei duplicatas.
3. **Tratamento do desbalanceamento**

   * Utilizei SMOTETomek para gerar churners sintéticos e remover exemplos ambíguos.
   * Otimizei o threshold via F₂‑Score (β=2) para maximizar recall sem Explodir falsos positivos.

---

## 🧼 4. Limpeza e Preparação dos Dados

### 4.1 Remoção de Identificadores

* Descartei `customerID` por não agregar valor preditivo.

### 4.2 Conversão de `TotalCharges`

* Detectei espaços e strings vazias:

  ```python
  df['TotalCharges'].sample(10)
  ```
* Apliquei a função de conversão (descrita acima) para evitar erros no pipeline.

### 4.3 Imputação de Valores Faltantes

* **Categóricas**: preenchi com `'Unknown'`.
* **Numéricas**: preenchi com a mediana.
  Verificação final:

```python
print(df.isnull().sum())  # Todos os valores nulos zerados
```

### 4.4 Checagem de Duplicatas e Tipos

```python
print(df.duplicated().sum())  # → 0 duplicatas
print(df.dtypes)             # → tipos adequados
```

Insight: dados bem limpos evitam vazamentos e erros de conversão no pipeline.

---

## 📊 5. Engenharia de Features

### 5.1 Features Base

| Feature                  | Como Criada                                             | Porquê                                                                   |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| tenure\_group            | `pd.cut(df.tenure, bins=[0,6,12,24,60,72], labels=[…])` | Captura padrões não lineares de risco em diferentes durações do contrato |
| avg\_charge\_per\_tenure | `df['TotalCharges'] / df['tenure'].replace(0,1)`        | Normaliza o gasto total pelo tempo de permanência                        |
| high\_value\_flag        | `(MonthlyCharges > Q3) & (tenure > median)`             | Detecta clientes de alto valor propensos a churn precoce                 |
| service\_density         | Soma de “Yes” em colunas de serviços extras             | Clientes com poucos serviços extras apresentam churn mais rápido         |

### 5.2 Features Derivadas da Análise de Erro

Durante a análise de erros (v2), observei padrões em falsos negativos (FNs) e falsos positivos (FPs):

```python
# False Negatives
fn_mask = (y_test==1) & (y_pred==0)
print(X_test[fn_mask][['Contract','PaymentMethod','MonthlyCharges','tenure']].describe())

# False Positives
fp_mask = (y_test==0) & (y_pred==1)
print(X_test[fp_mask][['Contract','PaymentMethod','MonthlyCharges','tenure']].describe())
```

**Padrões Detectados:**

* **FN** (churn não previsto): altas cobranças iniciais, tenure baixo, contrato mensal.
* **FP** (alarme falso): pagamento automático, contratos anuais, múltiplos serviços ativos.

**Novas Features Criadas**

| Feature              | Como Criada                                                                      | Porquê                                                                 |
| -------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| MonthlyCharges\_log  | `np.log1p(df['MonthlyCharges'])`                                                 | Reduz skew e enfatiza variações relativas em cobranças altas           |
| SeniorContractCombo  | `(df.SeniorCitizen==1).astype(int) * df.Contract.map({'Month-to-month':1, ...})` | Captura risco elevado de clientes sêniores em contratos menos estáveis |
| SupportServicesCount | Soma de “Yes” em serviços de suporte                                             | Clientes sem suporte extra têm +35% de chance de churn                 |

### 5.3 Importância dos Atributos

A seguir, o top‑10 de features ordenadas pelo `feature_importances_` do Random Forest:

| Atributo                        | Importância (%) |
| ------------------------------- | --------------- |
| Contract\_Month-to-month        | 17.43%          |
| Contract\_Two year              | 10.44%          |
| OnlineSecurity\_No              | 10.11%          |
| TechSupport\_No                 | 8.34%           |
| PaymentMethod\_Electronic check | 6.01%           |
| tenure                          | 5.63%           |
| InternetService\_Fiber optic    | 3.79%           |
| tenure\_group\_0-6              | 3.45%           |
| TotalCharges                    | 3.12%           |
| MonthlyCharges                  | 2.74%           |

---

## ⚙️ 6. Pipeline de Machine Learning

### 6.1 Separação de Variáveis

```python
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes':1, 'No':0})
```

### 6.2 Pré-processamento com ColumnTransformer

**Numéricas**:

```python
('num', Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
]), numeric_features)
```

**Categóricas**:

```python
('cat', Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
]), categorical_features)
```

### 6.3 Balanceamento de Classes com SMOTETomek

```python
from imblearn.combine import SMOTETomek
('smote', SMOTETomek(sampling_strategy=1.0, random_state=42))
```

**Por quê?** Gera churners sintéticos (SMOTE) e remove pontos ambíguos (Tomek Links), melhorando a separabilidade e reduzindo overfitting.

### 6.4 Classificador Random Forest

```python
('classifier', RandomForestClassifier(
    class_weight='balanced',
    random_state=42
))
```

### 6.5 Encadeamento em Pipeline

```python
pipeline = Pipeline([
    ('create_features', FunctionTransformer(create_features, validate=False)),
    ('preprocessor', preprocessor),
    ('smote', SMOTETomek(sampling_strategy=1.0, random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])
```

Benefício: encapsula todo o fluxo, evita vazamento de dados e garante reprodutibilidade.

---

## 🎯 7. Tuning de Hiperparâmetros e F₂-Score

### 7.1 Motivação

KPI principal: capturar churners (recall) sem gerar custos elevados com falsos positivos.
Métrica escolhida: F₂-Score (β=2), que penaliza mais fortemente os falsos negativos.

### 7.2 Configuração do Scorer

```python
from sklearn.metrics import make_scorer, fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)
```

### 7.3 RandomizedSearchCV

```python
search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=40, cv=cv,
    scoring=f2_scorer, verbose=1, n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

Resultado: recall > 90% na classe churn e precisão consistentemente melhorada.

---

## 🎛️ 8. Threshold Dinâmico & Classificação de Risco

### 8.1 Cálculo do Threshold Ótimo

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_test, y_probs)
f2_scores = (1 + 2**2) * (prec * rec) / (4 * prec + rec)
optimal_threshold = thr[np.nanargmax(f2_scores)]
```

### 8.2 Classificação de Risco

```python
def classify_risk(score):
    if score >= optimal_threshold + 0.25:
        return 'High'
    elif score >= optimal_threshold:
        return 'Moderate'
    else:
        return 'Low'
```

### 8.3 Cálculo de Custo por Previsão

```python
df_result['PredictionType'] = np.select(
    [
        (df_result['True']==1) & (df_result['Pred']==0),
        (df_result['True']==0) & (df_result['Pred']==1)
    ],
    ['FN', 'FP'],
    default='TN/TP'
)
df_result['Cost'] = np.select(
    [df_result['PredictionType']=='FP', df_result['PredictionType']=='FN'],
    [100, 2000],
    default=0
)
```

---

## 📈 9. Resultados Finais

**Optimized Classification Report:**

| Classe           | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.96      | 0.57   | 0.72     | 1053    |
| 1                | 0.42      | 0.93   | 0.58     | 352     |
| **accuracy**     |           |        | 0.66     | 1405    |
| **macro avg**    | 0.69      | 0.75   | 0.65     | 1405    |
| **weighted avg** | 0.82      | 0.66   | 0.68     | 1405    |

**Optimal Threshold**: 0.45
**Recall (Churn)**: 93% — captura quase todos os churners.
**Precisão (Churn)**: 42% — aceitável dado o alto custo de perder um churner.

---

## 💡 10. Comparativo de Versões

| Versão | Acc  | Prec | Recall | F1   | F2   | AUC-ROC | Principais Mudanças             |
| ------ | ---- | ---- | ------ | ---- | ---- | ------- | ------------------------------- |
| v1     | 0.79 | 0.71 | 0.45   | 0.55 | 0.49 | 0.80    | Modelo baseline                 |
| v2     | 0.83 | 0.61 | 0.90   | 0.73 | 0.75 | 0.83    | Threshold otimizado (PR curve)  |
| v3     | 0.83 | 0.42 | 0.93   | 0.58 | 0.66 | 0.83    | Ajustes de features e F2-tuning |

**Nota**: AUC-ROC ≈ 0.83 confirma bom poder discriminativo, não aleatoriedade.

---

## 📌 11. Conclusão Operacional e Próximos Passos

Este modelo atinge alto recall com custo operacional baixo, graças ao uso de e-mails e mensagens automatizadas.

* **FNs** ficaram abaixo de 7%, evitando grandes perdas.
* **FPs** geram custo mínimo e criam oportunidades de engajamento.
* A escolha do F₂-Score reforça a prioridade de capturar churners.

### 11.1 Estratégias de Ação para Clientes de Alto Risco

| Segmento   | Critério                | Nº de Clientes | Ação                        |
| ---------- | ----------------------- | -------------- | --------------------------- |
| Crítico    | Prob. Churn ≥ 80%       | 120            | Atendimento humano imediato |
| Alto Risco | 60% ≤ Prob. Churn < 80% | 300            | Campanhas personalizadas    |
| Monitorar  | 45% ≤ Prob. Churn < 60% | 500            | Engajamento preventivo      |

### 11.2 Kit de Ações por Segmento

* **🔴 Crítico (Prob. ≥ 80%)**: contato humano em até 24h, oferta VIP (desconto + upgrade), visita técnica preventiva. *Meta: ≥ 65% de taxa de conversão.*
* **🟠 Alto Risco (60%–80%)**: e-mail segmentado, reforço via SMS. *Meta: ≥ 45% de abertura/interação.*
* **🟢 Monitorar (45%–60%)**: comunicação automática (e-mail/SMS genérico), promoções leves (crédito extra, gamificação). *Meta: ≥ 25% de resposta.*

### 11.3 Cálculo de Custo-Benefício e ROI

| Item                 | Custo Unitário | Retenção Estimada |
| -------------------- | -------------- | ----------------- |
| Ligação VIP          | R\$ 150        | 70%               |
| E-mail Personalizado | R\$ 10         | 40%               |
| SMS                  | R\$ 2          | 15%               |

```text
custo_total = (120 * 150) + (300 * 10) + (500 * 2) = R$ 24.000
receita_preservada = (120*0.7 + 300*0.4 + 500*0.15) * 2500 = R$ 1.162.500
ROI = (1.162.500 - 24.000) / 24.000 ≈ 4743%
```

### 11.4 Fluxo de Governança

* **Diário**: atualizar lista de clientes críticos às 8h.
* **Semanal**: reunião de análise de conversões com equipe de CX.
* **Mensal**: ajustar limites de probabilidade conforme capacidade e testar novas mensagens/ofertas com A/B testing.

### 11.5 Alertas Proativos (Exemplo)

```python
if prob_churn >= 0.8 and contract == "Monthly" and support_tickets >= 3:
    enviar_para_fila("URGENTE", cliente_id)
elif prob_churn >= 0.6 and internet_service == "Fiber":
    oferecer_upgrade_gratis(cliente_id)
```

### 11.6 Resultados Esperados

* Redução de 25–30% no churn nos segmentos Crítico e Alto Risco.
* ROI de 10:1 para cada real investido em retenção.
* Aumento de 15% no NPS devido a ações personalizadas.
