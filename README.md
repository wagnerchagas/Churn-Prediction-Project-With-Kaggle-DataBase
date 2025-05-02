# 🧠 Projeto de Previsão de Churn em Telecomunicações- telco-costumer-churn Kaggle

## LINK PARA VIZUALIZAÇÃO, DASHBOARD FEITO COM STREAMLIT:https://churn-prediction-project-with-kaggle-database-g5q9uswffyie7tzb.streamlit.app/
## IMAGENS ESTÃO CONTIDAS NA PASTA "prints do funcionamento", onde estão diversos prints das funcionalidades do código, em estágio inicial e final.

## 📉 Introdução
Este projeto tem como objetivo prever o churn de clientes de uma operadora de telecomunicações, utilizando o conjunto de dados Telco Customer Churn.
Através de técnicas de aprendizado de máquina e análise exploratória, concentrei meus esforços em identificar os principais fatores que influenciam a saída dos clientes, 
desenvolver um modelo preditivo eficaz e propor estratégias de retenção com base em perfis de risco. Além disso, incorporei uma análise de custo-benefício para ajudar na tomada de-
decisão orientada por dados.

**Origem dos Dados**: Telco Customer Churn (Kaggle)  
**Técnicas Principais**:  
1. **Random Forest**  
   - Escolhido por sua robustez e facilidade de interpretação via importância de atributos.  
2. **SMOTETomek (SMOTE + Tomek Links)**  
   - **SMOTE** gera amostras sintéticas da classe minoritária.  
   - **Tomek Links** remove pares de pontos ambíguos na fronteira de decisão.  
3. **Engenharia de Features Avançada**  
   - Variáveis criadas a partir de padrões apontados pela análise de erros (e.g. `tenure_group`, `high_value_flag`, `SeniorContractCombo`, `SupportServicesCount`).  
4. **Otimização de Threshold via F2-Score**  
   - F2 prioriza o **recall** (capturar churners) em relação à precisão.
5. **Pré-processamento e Pipeline Unificado**
   -Utiliza ColumnTransformer para imputação e escala separadas em variáveis numéricas e codificação em variáveis categóricas, tudo encadeado em um único Pipeline sem vazamento de dados.
6. **Hyperparameter Tuning com RandomizedSearchCV**
   -Busca aleatória em um espaço de parâmetros (n_estimators, max_depth, class_weight, smote__sampling_strategy etc.) usando make_scorer(fbeta_score, beta=2) para otimizar o F2-Score.

---

## 1. 🎯 Objetivo do Projeto
Prever churn de clientes com alta sensibilidade (recall) e precisão suficiente para que a equipe de retenção:
1. Concentre esforços nos casos de maior risco.  
2. Minimize desperdício de recursos em falsos alarmes.  

---

### 2. 🔍 Entendimento e Iniciação

### 2.1 Escolha do Dataset
- **Fonte**: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Conteúdo**:  
  - Dados demográficos (gênero, idade, dependentes)  
  - Tipo de contrato (`Contract`), método de pagamento  
  - Uso de serviços (TV, internet, suporte técnico)  
  - Métricas de consumo (`MonthlyCharges`, `TotalCharges`)  
  - Flag `Churn` (“Yes”/“No”)  

### 2.2 Perguntas de Negócio
- Quais atributos realmente influenciam a decisão de churn?  
- Como limpar inconsistências (e.g., `TotalCharges` como texto, espaços vazios)?  
- Em um dataset desbalanceado (~27% churn), como priorizar a detecção de churners sem gerar alarmes falsos em excesso?

### 2.3 Respostas às Perguntas de Negócio
1. **Atributos que influenciam o churn**  
   – Vou extrair `feature_importances_` do Random Forest e mostrar um gráfico ou tabela com os 10 mais importantes (e.g. `PaymentMethod_Electronic check`, `Contract_Month-to-month`, etc.).
2. **Limpeza de inconsistências**  
   – Convertemos `TotalCharges` de string para numérico com `pd.to_numeric(..., errors='coerce')` + `fillna(0)`.  
   – Imputação de valores faltantes em categóricas (`'Unknown'`) e numéricas (mediana).  
   – Remoção de identificadores e checagem de duplicatas.
3. **Tratamento do desbalanceamento**  
   – Usamos SMOTETomek para gerar churners sintéticos e remover exemplos ambíguos.  
   – Otimizamos o threshold via F₂‑Score (β=2) para maximizar o recall sem explodir os falsos positivos.

---

---

## 3. 🧼 Limpeza e Preparação dos Dados

1. **Remoção de Identificadores**  
   - `customerID` não aporta valor preditivo e foi descartado.  

2. **Conversão de `TotalCharges`**  
   - Identificamos espaços e strings vazias:  
     ```python
     df['TotalCharges'].sample(10)
     ```  
   - Função de conversão:
     ```python
     def convert_total_charges(s):
         """Convert TotalCharges to numeric, handling empty strings and spaces."""
         return pd.to_numeric(
             s.astype(str).str.strip().replace('', np.nan),
             errors='coerce'
         ).fillna(0)
     df['TotalCharges'] = convert_total_charges(df['TotalCharges'])
     ```
   - **Por quê?** Evitamos que strings vazias quebrem o pipeline e garantimos valores numéricos utilizáveis.

3. **Imputação de Valores Faltantes**  
   - **Categóricas**: preenchidas com `'Unknown'`  
   - **Numéricas**: preenchidas com a **mediana**  
   - Verificação final:
     ```python
     print(df.isnull().sum())  # Todos os valores nulos zerados
     ```

4. **Checagem de Duplicatas e Tipos**  
   ```python
   print(df.duplicated().sum())  # → 0 duplicatas
   print(df.dtypes)             # → tipos adequados
Insight: dados bem limpos evitam vazamentos e erros de conversão no pipeline.

### 4. 📊 Engenharia de Features
### 4.1 Features Base

Feature	Como Criada e 	Porquê
tenure_group	pd.cut(df.tenure, bins=[0,6,12,24,60,72], labels=[…])	Captura padrões não lineares de risco em diferentes durações de contrato
avg_charge_per_tenure	df['TotalCharges'] / df['tenure'].replace(0,1)	Normaliza o gasto total pelo tempo de permanência
high_value_flag	(MonthlyCharges > Q3) & (tenure > median)	Detecta clientes de alto valor propensos a churn precoce
service_density	Soma de respostas “Yes” em colunas de serviços extras (OnlineSecurity, TechSupport, etc.)	Clientes com poucos serviços extras apresentaram churn mais rápido

### 4.2 Features Derivadas da Análise de Erro
Durante os testes iniciais (v2), observamos discrepâncias em Falsos Negativos (FNs) e Falsos Positivos (FPs):

### 4.3 Importância dos Atributos

A seguir, o top‑10 de features ordenadas pelo `feature_importances_` do Random Forest:

| Atributo                        | Importância (%) |
|---------------------------------|-----------------|
| Contract_Month-to-month         | 17.43%          |
| Contract_Two year               | 10.44%          |
| OnlineSecurity_No               | 10.11%          |
| TechSupport_No                  |  8.34%          |
| PaymentMethod_Electronic check  |  6.01%          |
| tenure                          |  5.63%          |
| InternetService_Fiber optic     |  3.79%          |
| tenure_group_0-6                |  3.45%          |
| TotalCharges                    |  3.12%          |
| MonthlyCharges                  |  2.74%          |


python

# False Negatives
fn_mask = (y_test==1) & (y_pred==0)
print(X_test[fn_mask][['Contract','PaymentMethod','MonthlyCharges','tenure']].describe())

# False Positives
fp_mask = (y_test==0) & (y_pred==1)
print(X_test[fp_mask][['Contract','PaymentMethod','MonthlyCharges','tenure']].describe())
Padrão Detectado	Observação
FN (churn não previsto)	Altas cobranças iniciais, tenure baixo, contrato mensal
FP (alarme falso)	Pagamento automático, contratos anuais, múltiplos serviços ativos

## Novas Features Criadas

## Feature	Como Criada	Porquê
MonthlyCharges_log	np.log1p(df['MonthlyCharges'])	Reduz skew e enfatiza variações relativas em cobranças altas
SeniorContractCombo	(df.SeniorCitizen==1).astype(int) * df.Contract.map({'Month-to-month':1, ...})	Captura risco elevado em clientes sêniores com contratos menos estáveis
SupportServicesCount	Soma de respostas “Yes” em serviços de suporte	Clientes sem suporte extra têm +35% de chance de churn

## 5. ⚙️ Pipeline de Machine Learning
## 5.1 Separação de Variáveis
python

X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes':1, 'No':0})
5.2 Pré-processamento com ColumnTransformer
Numéricas:

python

('num', Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
]), numeric_features)
Categóricas:

python

('cat', Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
]), categorical_features)

## 5.3 Balanceamento de Classes com SMOTETomek
python

from imblearn.combine import SMOTETomek

('smote', SMOTETomek(sampling_strategy=1.0, random_state=42))
Por quê?

Gera churners sintéticos (SMOTE)

Remove pontos ambíguos (Tomek Links)

Melhora a separabilidade e reduz overfitting

## 5.4 Classificador Random Forest
python

('classifier', RandomForestClassifier(
    class_weight='balanced',
    random_state=42
))
## 5.5 Encadeamento em Pipeline
python

pipeline = Pipeline([
    ('create_features', FunctionTransformer(create_features, validate=False)),
    ('preprocessor', preprocessor),
    ('smote', SMOTETomek(sampling_strategy=1.0, random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])
Benefício: encapsula todo o fluxo, evitando vazamento de dados e garantindo reprodutibilidade.

## 6. 🎯 Tuning de Hiperparâmetros e F2-Score
6.1 Motivação
KPI principal: capturar churners (recall) sem gerar custos altos com falsos positivos.
Métrica escolhida: F2-Score (β=2), que penaliza mais fortemente os falsos negativos.

## 6.2 Configuração do Scorer
python
from sklearn.metrics import make_scorer, fbeta_score
f2_scorer = make_scorer(fbeta_score, beta=2)

## 6.3 RandomizedSearchCV
python

search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=40, cv=cv,
    scoring=f2_scorer, verbose=1, n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
Resultado: recall > 90% na classe churn e precisão consistentemente melhorada.

## 7. 🎛️ Threshold Dinâmico & Classificação de Risco
## 7.1 Cálculo do Threshold Ótimo
python

from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y_test, y_probs)
f2_scores = (1 + 2**2) * (prec * rec) / (4 * prec + rec)
optimal_threshold = thr[np.nanargmax(f2_scores)]

## 7.2 Classificação de Risco
python

def classify_risk(score):
    """Classify churn risk into High, Moderate, Low based on probability score."""
    if score >= optimal_threshold + 0.25:
        return 'High'
    elif score >= optimal_threshold:
        return 'Moderate'
    else:
        return 'Low'
        
## 7.3 Cálculo de Custo por Previsão
python

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
    [100, 2000],  # custo FP=R$100, FN=R$2.000
    default=0
)

## 7.4 Exportação Final
python

df_result.to_csv('clientes_classificados.csv', index=False, sep=';')
print(df_result.head(20))  # Visual check in terminal
7.5 Trade-off: Recall Alto vs Precisão Baixa
Recall (Churn) = 93% → captura quase TODOS os churners (326 de 350).

Precisão (Churn) = 42% → 461 FPs, muitos falsos alarmes.

Quando isso faz sentido?

✅ Se o custo de perder um cliente for muito alto e as ações de retenção forem automatizadas e baratas.

❌ Se há pouca capacidade operacional para lidar com falsos positivos ou campanhas caras.

## 7.6 Impacto de Falsos Positivos e Negativos no Negócio
FPs: custo irrelevante (e-mails automáticos → quase zero); só gera custo se o cliente responder.

FNs: apenas 24 (< 7% dos churners) → excelente para evitar perdas sem alerta.

Estratégias de Ação

Escalonamento de Retenção

Score > 0.70 → ações personalizadas

0.48–0.70 → e-mails automáticos

Qualidade das Mensagens

Evitar genéricos (“Não vá embora!”)

Preferir sutis (“Conteúdo exclusivo para você!”)

Monitoramento Contínuo

Se muitos FPs responderem → sinal real de insatisfação

Se poucas respostas → revisar segmentação

## 8. 📈 Resultados Finais

Optimized Classification Report:
              precision    recall  f1-score   support
           0       0.96      0.57      0.72      1053
           1       0.42      0.93      0.58       352

    accuracy                           0.66      1405
   macro avg       0.69      0.75      0.65      1405
weighted avg       0.82      0.66      0.68      1405

Optimal Threshold: 0.45

Recall (Churn): 93% — captura quase todos os churners.
Precisão (Churn): 42% — aceitável dado o alto custo de perder um churner.

## 9. 💡 Comparativo de Versões
Versão	Acc	Prec	Recall	F1	F2	AUC-ROC	Principais Mudanças
v1	0.79	0.71	0.45	0.55	0.49	0.80	Modelo baseline
v2	0.83	0.61	0.90	0.73	0.75	0.83	Threshold otimizado (PR curve)
v3	0.83	0.42	0.93	0.58	0.66	0.83	Ajustes de features e F2-tuning

Nota: AUC-ROC ≈ 0.83 confirma bom poder discriminativo, não aleatoriedade.


## 10. 📌 Conclusão Operacional
O modelo atinge o objetivo estratégico de priorizar recall, mas precisa ser acompanhado de:


 Conclusão Operacional: Estratégias de Ação para Clientes de Alto Risco
Objetivo:
Reduzir o churn em 25% nos próximos 3 meses através de intervenções direcionadas.

------------------------------------------------------------
## 10.1 Triagem de Clientes Prioritários

Segmento      | Critério                    | Nº de Clientes | Ação
--------------|-----------------------------|----------------|----------------------------
Crítico       | Prob. Churn ≥ 80%           | 120            | Atendimento humano imediato
Alto Risco    | 60% ≤ Prob. Churn < 80%     | 300            | Campanhas personalizadas
Monitorar     | 45% ≤ Prob. Churn < 60%     | 500            | Engajamento preventivo

------------------------------------------------------------
## 10.2 Kit de Ações para Cada Segmento

🔴 Crítico (Prob. ≥ 80%)
- Contato humano em até 24h (call center especializado)
- Oferta VIP: desconto + upgrade de serviço
- Visita técnica preventiva, se aplicável
> Métrica-chave: ≥ 65% de taxa de conversão

🟠 Alto Risco (60% ≤ Prob. < 80%)
- Campanha de e-mail segmentada com oferta relevante
- Reforço via SMS
> Métrica-chave: ≥ 45% de taxa de abertura/interação

🟢 Monitorar (45% ≤ Prob. < 60%)
- Comunicação automática (e-mail/SMS genérico)
- Promoções leves, como crédito extra ou gamificação
> Métrica-chave: ≥ 25% de taxa de resposta

------------------------------------------------------------
## 10.3 Cálculo de Custo-Benefício

Item                | Custo Unitário | Retenção Estimada
--------------------|----------------|-------------------
Ligação VIP         | R$ 150         | 70%
E-mail Personalizado| R$ 10          | 40%
SMS                 | R$ 2           | 15%

Exemplo de ROI:

custo_total = (120 * 150) + (300 * 10) + (500 * 2) = R$ 24.000
receita_preservada = (120*0.7 + 300*0.4 + 500*0.15) * 2500 = R$ 1.162.500
ROI = (1.162.500 - 24.000) / 24.000 ≈ 4743%

------------------------------------------------------------
## 10.4 Fluxo de Governança

- Diário: Atualizar lista de clientes críticos às 8h
- Semanal: Reunião de análise de conversões com a equipe de CX
- Mensal:
  - Ajustar limites de probabilidade conforme capacidade
  - Testar novas mensagens/ofertas com A/B testing

------------------------------------------------------------
## 10.5 Alertas Proativos (Exemplo)

if prob_churn >= 0.8 and contract == "Monthly" and support_tickets >= 3:
    enviar_para_fila("URGENTE", cliente_id)
elif prob_churn >= 0.6 and internet_service == "Fiber":
    oferecer_upgrade_gratis(cliente_id)

------------------------------------------------------------
Resultado Esperado:

- Redução de 25–30% no churn nos segmentos Crítico e Alto Risco
- ROI de 10:1 para cada real investido em retenção
- Aumento de 15% no NPS devido a ações personalizadas


## 10.6📌 Conclusão e Próximos Passos
O modelo atinge alto recall com custo operacional baixo, graças ao uso de e-mails e mensagens automatizadas.

FNs são mantidos abaixo de 7%;

FPs geram custo mínimo e criam oportunidades de engajamento.

F2-Score como métrica principal reforça a prioridade de capturar churners.
