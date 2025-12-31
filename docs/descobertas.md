# Descobertas da exploração

## Insights
### Sobre os dados
- Após limpeza e tratamento, os anos de 2015 e 2016 ficaram sem registros, e 2017 ficou com apenas 190 exemplos. Por esse motivo, eles foram desconsiderados na análise, que se concentrou nos anos de 2018 a 2024.
- No geral, as variáveis numéricas possuem distribuições diferentes entre si. As mais concentradas são `frp`, `precipitacao_max` e `num_deteccoes_dia` e, as com mais outliers, `frp`, `precipitacao_max` e `num_dias_sem_chuva_max`.
- Correlações das variáveis numéricas mais relevantes:
  - `hora_primeira_deteccao` e `hora_ultima_deteccao`:  0,97
  - `num_dias_sem_chuva_max` e `risco_fogo_max`:  0,42
  - `longitude` e `risco_fogo_max`:  0,33
  - `precipitacao_max` e `risco_fogo_max`: -0,36
- A análise de correlação das variáveis categóricas mostrou muitos relacionamentos "óbvios", como mês com trimestre ou dia do ano com semana do ano.
- Além de `bioma` (que foram excluídos), as variáveis com dados faltantes foram somente de dados climáticos (`num_dias_sem_chuva_max`, `precipitacao_max` e `risco_fogo_max`). Eles foram imputados a partir da seguinte estratégia:
  1. mediana baseada na região geográfica próxima ou;
  2. mediana por bioma;
  3. mediana global.

### Sobre o fenômeno
- A Amazônia é o bioma com maior número de focos de queimadas, representando 46,5% dos registros.
- Os estados do Pará (18,4%) e Mato Grosso (15,7%) são os que mais apresentam focos de queimadas.
- Os meses de agosto (21,3%) e setembro (29,5%) concentram a maioria dos registros.
- O 3.º trimestre do ano (57%) é o período com mais focos de queimadas.
- A maioria dos focos de queimadas ocorre à tarde (73,4%).
- A primavera (57,7%) é a estação do ano com mais registros de queimadas.
- O período seco (73,3%) é o que apresenta a maior quantidade de focos de queimadas.

## Limitações
- Há um desbalanceamento na quantidade de registros por ano e por bioma, o que deve ser levado em consideração na modelagem.

## Sugestões para próximos passos
- Para uma tarefa de regressão, recomenda-se o uso do `df_final_regression`, que contém a variável alvo FRP (potência radiativa do fogo).
- Para uma tarefa de classificação, recomenda-se o uso do `df_final_classification`, que contém a variável alvo `label_intensidade` (baixa, média e alta).
- Explorar modelos de séries temporais para capturar padrões sazonais e tendências ao longo dos anos.
- Elaborar visualizações para identificar a distribuição espacial e temporal dos focos de queimadas, hotspots e possíveis áreas de incêndios emergentes.