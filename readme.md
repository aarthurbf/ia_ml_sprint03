# Ford — Segmentação e Classificação de Clientes para Retenção na Rede Oficial

Projeto desenvolvido para o Sprint de Inteligência Artificial & Machine Learning, propondo uma solução analítica para o desafio da Ford: prever, já no momento da compra do veículo, qual será o comportamento futuro do cliente em relação à manutenção na rede oficial, permitindo ações de retenção proativas e personalizadas.

## Equipe

| Nome | RM |
|---|---|
| Arthur Bobadilla Franchi | 555056 |
| Luan Orlandelli Ramos | 554747 |
| Jorge Luiz | 554418 |

## Contexto do problema

O veículo é apenas o ponto de entrada do relacionamento com o cliente Ford. O ciclo de vida completo revisões periódicas, troca de peças e manutenções preventivas é a principal fonte de receita recorrente da rede autorizada. Hoje a concessionária só descobre se um cliente vai ser fiel à rede oficial depois de observar seu comportamento por meses, quando já é tarde para agir. Este projeto resolve essa assimetria de informação.

## Abordagem: solução em duas etapas

Para evitar contaminação do modelo preditivo com informação futura (anti-data leakage), o projeto foi estruturado em dois estágios complementares:

```
ETAPA 1 — SEGMENTAÇÃO                    ETAPA 2 — CLASSIFICAÇÃO
Base 1 (Histórico Completo, 24m)   --->  Base 2 (Momento da Compra)
K-Means Clustering (k=4)                 Logistic Regression
Perfis Comportamentais Reais             Previsão do Perfil Futuro
```

- **Etapa 1 (não supervisionada):** aplica K-Means sobre 12 variáveis comportamentais pós-venda para descobrir os perfis reais de comportamento dos clientes (`fiel`, `econômico`, `esquecido`, `abandono`).
- **Etapa 2 (supervisionada):** transforma os rótulos dos clusters em variável-alvo e treina um classificador usando exclusivamente as variáveis disponíveis no momento da compra, simulando o cenário real de operação da concessionária.

## Dados

- **Base 1 (Histórico completo):** 500.000 registros, 37 variáveis, 0,4% de nulos. Usada na Etapa 1 (segmentação).
- **Base 2 (Momento da compra):** 500.000 registros, 24 variáveis, 1,2% de nulos. Usada na Etapa 2 (classificação), contendo apenas dados conhecidos no ato da venda.

> Os notebooks esperam o upload interativo via `google.colab.files.upload()`.

## Preparação dos dados

| Problema | Decisão | Justificativa |
|---|---|---|
| Valores nulos (< 2,5% por coluna) | Imputação por mediana | Robusta a outliers; taxa baixa não distorce a distribuição |
| 330 registros com `compra_a_vista=0` e `prazo=0` | Corrigido para `compra_a_vista=1` | Inconsistência lógica: sem prazo de financiamento = pagamento à vista |
| Outliers em renda, distância e gasto | Winsorização 1%–99% | Preserva a distribuição real, elimina extremos espúrios |
| Variáveis categóricas (5 colunas) | One-Hot Encoding | Compatibilidade com algoritmos baseados em distância/gradiente |
| Escalas heterogêneas | `StandardScaler` | K-Means e Logistic Regression são sensíveis à magnitude das variáveis |

## Modelos avaliados

| Modelo | Tipo | Papel |
|---|---|---|
| **Logistic Regression** | Linear | Modelo eleito para produção |
| Random Forest | Ensemble | Comparativo / feature importance |
| Decision Tree | Não-linear | Baseline |
| MLP | Não-linear | Teste de ganho por interações não-lineares |

Também foram realizados: **GridSearchCV** (ajuste de `C` e `penalty` da Logistic Regression) e **validação cruzada** (5 folds estratificados) para garantir robustez do modelo final.

## Resultados

### Segmentação (Etapa 1 — K-Means, k=4)

| Segmento | % Base | Revisões/24m | Share Rede | Gasto (24m) | Satisfação | Churn |
|---|---|---|---|---|---|---|
| 🟢 Fiel | 30,1% | 3,95 | 93,1% | R$ 4.266 | 9,4/10 | 2,2% |
| 🟠 Econômico | 25,4% | 2,73 | 57,4% | R$ 1.429 | 7,1/10 | 7,1% |
| 🔴 Esquecido | 23,8% | 1,13 | 26,2% | R$ 315 | 3,9/10 | 25,3% |
| 🔵 Abandono | 20,7% | 1,23 | 17,4% | R$ 213 | 4,8/10 | 27,5% |

Silhouette Score ≈ 0,32; concordância ≥ 91% com o ground truth (`perfil_latente`).

### Classificação (Etapa 2 — comparação de algoritmos)

| Modelo | Acurácia | F1 Macro | F1 Weighted | Decisão |
|---|---|---|---|---|
| **Logistic Regression** | **79,0%** | **0,766** | **0,786** | **Eleito** |
| Random Forest | 78,8% | 0,768 | 0,785 | Comparativo |
| MLP (Rede Neural) | ≈78% | ≈0,74 | ≈0,78 | Comparativo |
| Decision Tree | 74,3% | 0,741 | 0,742 | Baseline |

### Métricas por classe — Logistic Regression (modelo final)

| Segmento | Precisão | Recall | F1-Score |
|---|---|---|---|
| Fiel | 0,97 | 0,98 | 0,97 |
| Econômico | 0,80 | 0,86 | 0,83 |
| Esquecido | 0,69 | 0,72 | 0,71 |
| Abandono | 0,61 | 0,52 | 0,56 |

A confusão entre **Esquecido** e **Abandono** é estrutural (ambos com baixo planejamento de manutenção no ato da compra) e aceitável do ponto de vista de negócio: classificar um pelo outro ainda aciona o protocolo correto de intervenção de alto risco.

### Top 5 drivers preditivos

1. `organizacao_proxy` (0,25) — comportamento organizado no ato da compra prediz o perfil Fiel
2. `sensibilidade_preco_inicial` (0,25) — alta sensibilidade discrimina Econômico e Abandono
3. `tempo_decisao_dias` (0,08) — decisão rápida correlaciona com Abandono
4. `distancia_concessionaria_km` (0,08) — maior distância eleva risco de evasão
5. `score_credito` (0,05) — score mais alto associado ao perfil Fiel

## Modelo selecionado

**Logistic Regression** foi eleita como solução final por:

- Melhor acurácia geral e F1 Weighted entre os quatro algoritmos testados;
- Interpretação nativa dos coeficientes por classe, auditável pelo time de negócio;
- Inferência em tempo real, sem custo computacional relevante — ideal para integração no CRM;
- Estabilidade confirmada por validação cruzada.

## Proposta de deploy

1. Serializar o modelo (`LogisticRegression` + `StandardScaler`) via `joblib`.
2. Empacotar como microsserviço REST (ex.: FastAPI) com endpoint `POST /predict-segmento`.
3. Integrar ao CRM da concessionária: no fechamento da venda, o sistema chama o endpoint e grava o segmento previsto, disparando o protocolo de retenção correspondente (Fiel, Econômico, Esquecido ou Abandono).
4. Monitorar *data drift* e *concept drift*, comparando periodicamente a previsão com o comportamento real observado após 24 meses.
5. Retreinar o modelo periodicamente (trimestral/semestral) com novos clientes que já completaram o ciclo de relacionamento.

## Melhorias e trabalhos futuros

- Calibração de probabilidades (`CalibratedClassifierCV`) para priorização de campanhas;
- Engenharia de features adicionais (histórico multimarca, sazonalidade, geolocalização mais granular);
- Classificador binário dedicado à fronteira Esquecido/Abandono;
- Ensembles/stacking combinando Logistic Regression e Random Forest;
- Coleta de sinais comportamentais de curto prazo (4–8 semanas pós-compra) para re-classificação antecipada;
- Teste A/B em campo para validar o ROI real das ações de retenção.

## Como executar

Os notebooks foram desenvolvidos para o Google Colab (`google.colab.files.upload()`), mas rodam em qualquer ambiente Jupyter com pequenas adaptações:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

> Rode os notebooks de cima para baixo, sem pular células — algumas células dependem de variáveis definidas em células anteriores.

## Tecnologias utilizadas

`Python` · `pandas` · `numpy` · `scikit-learn` (K-Means, Logistic Regression, Random Forest, Decision Tree, MLPClassifier, GridSearchCV) · `matplotlib` · `seaborn`
