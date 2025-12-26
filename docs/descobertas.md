# Descobertas da exploração

## Insights
### Sobre os dados
- Os dados já estavam previamente limpos, sem valores nulos ou inconsistentes, assim como descrito na fonte.
- Nos anos de 2015 a 2022, as colunas 'satelite', 'dias_sem_chuva', 'precipitacao', 'risco_fogo' e 'potencia_radiativa_fogo' estão ausentes na sua totalidade. Elas estão presentes somente a partir de 2023. Dessa forma, podemos inferir que esses dados não eram coletados até então.
- Foram separados dois datasets diferentes: ambos com dados diários agregados de 2015 a 2024, mas um deles (`df_daily_stp`) contendo apenas os registros de 2023 e 2024, quando os dados meteorológicos começaram a ser coletados.
- No geral, as variáveis numéricas possuem distribuições diferentes entre si e assimétricas. As que possuem as suas distribuições mais concentradas são 'num_deteccoes_dia' e 'multiplas_deteccoes', enquanto a primeira é a que possui a maior quantidade outliers.
- As features numéricas possuem correlações fracas entre si. Contudo, as que merecem destaque são: 'hora_primeira_deteccao' e 'hora_ultima_deteccao' (0,99) e 'num_deteccoes_dia' e 'multiplas_deteccoes' (0,58).
- A análise de correlação das variáveis categóricas mostrou muitos relacionamentos "óbvios", como mês com trimestre ou dia do ano com semana do ano.
- Os dados do `df_daily_stp` foram imputados a partir da seguinte estratégia:
  1. mediana baseada na região geográfica próxima ou;
  2. mediana por bioma;
  3. mediana global.
- A maior proporção de dados faltantes do `df_daily_stp` pertence aos dados do ano de 2023 (3,4%). Além disso, as regiões com maior quantidade de dados ausentes são o Norte, o Nordeste e o Centro-Oeste. Por bioma, A Amazônia (2,2%) e o Cerrado (2,1%) lideram.

### Sobre o fenômeno
- A Amazônia é o bioma com maior número de focos de queimadas, representando 51,7% dos registros.
- Os estados do PA (22%) e MT (17,6%) são os que mais apresentam focos de queimadas.
- Os meses de agosto (22,4%) e setembro (29,6%) concentram a maioria dos registros.
- O 3.º trimestre do ano (58%) é o período com mais focos de queimadas.
- A maioria dos focos de queimadas ocorre à tarde (58,5%).
- A primavera (56,2%) é a estação do ano com mais registros de queimadas.
- O período seco (74,5%) é o que apresenta a maior quantidade de focos de queimadas.

## Limitações
- Não podemos aproveitar os dados meteorológicos coletados, pois só estão presentes a partir de 2023. Se os usássemos com os dados de anos anteriores, teríamos uma inconsistência de informações que comprometeria a análise. Por conta disso, decidimos remover as colunas relacionadas a esses dados, evitando ambas a inconsistência e a perda de um grande volume de registros.
- O dataset **não** permite inferência causal nem previsão de intensidade/PRF (potência radioativa do fogo) histórica.

## Sugestões para próximos passos
- Realizar uma análise temporal mais aprofundada, investigando tendências e sazonalidades ao longo dos anos.
- Buscar identificar hotspots e possíveis áreas emergentes de queimadas.