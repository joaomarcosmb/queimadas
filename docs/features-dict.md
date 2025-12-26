# Dicionário de features

## `df_daily`
Este dataset contém os registros de focos de queimadas agrupados por ponto geográfico e dia, isto é, os múltiplos registros por dia originais foram agregados num só. Os critérios de agregação podem ser vistos com mais detalhes no notebook `notebooks/1.0.0-eda-e-tratamento`.

**Tamanho**: 11.912.771 linhas x 18 colunas.

| Feature                  | Tipo        | Descrição                                                                    |
|--------------------------|-------------|------------------------------------------------------------------------------|
| latitude                 | Float64     | Latitude do ponto geográfico do registro                                     |
| longitude                | Float64     | Longitude do ponto geográfico do registro                                    |
| data_dia                 | Date        | Data do evento                                                               |
| bioma                    | Categorical | Bioma associado ao ponto geográfico                                          |
| sigla_uf                 | Categorical | Unidade da federação (UF)                                                    |
| ano                      | Int64       | Ano do evento                                                                |
| mes                      | Categorical | Mês do ano (1 – 12)                                                          |
| dia_ano                  | Categorical | Dia do ano (1 – 365/366)                                                     |
| dia_semana               | Categorical | Dia da semana do evento                                                      |
| semana_ano               | Categorical | Semana do ano (1 – 52/53)                                                    |
| trimestre                | Categorical | Trimestre do ano (1 – 4)                                                     |
| periodo_dia_predominante | Categorical | Período do dia predominante das detecções (madrugada, manhã, tarde ou noite) |
| estacao_ano              | Categorical | Estação do ano (verão, outono, inverno ou primavera)                         |
| periodo_climatico        | Categorical | Classificação climática sazonal (seco ou chuvoso)                            |
| hora_primeira_deteccao   | Int8        | Hora (0 – 23) da primeira detecção no dia                                    |
| hora_ultima_deteccao     | Int8        | Hora (0 – 23) da última detecção no dia                                      |
| num_deteccoes_dia        | UInt32      | Número total de detecções registradas no dia para o ponto                    |
| multiplas_deteccoes      | Int8        | Indicador binário (0/1) se houve mais de uma detecção no dia                 |

> **Importante**: Algumas colunas de tempo foram tratadas como categóricas, dada a sua natureza cíclica/sazonal.

## `df_monthly`
Este dataframe faz um resumo geral por mês, contendo métricas agregadas dos registros de incêndios.

**Tamanho**: 120 linhas x 11 colunas.

| Feature                   | Tipo    | Descrição                                                 |
|---------------------------|---------|-----------------------------------------------------------|
| mes_inicio                | Date    | Data correspondente ao primeiro dia do mês                |
| ano_mes_year              | Int32   | Ano de referência                                         |
| ano_mes_month             | Int8    | Mês de referência (1 – 12)                                |
| ano_mes                   | String  | Identificador textual do mês no formato `YYYY-MM`         |
| n_dias_com_evento         | UInt32  | Número de dias no mês com ao menos um evento registrado   |
| total_deteccoes_mes       | UInt32  | Total de detecções acumuladas no mês                      |
| media_deteccoes_por_dia   | Float64 | Média de detecções por dia (considerando dias com evento) |
| mediana_deteccoes_por_dia | Float64 | Mediana de detecções por dia no mês                       |
| pct_multiplas_deteccoes   | Float64 | Proporção de dias com múltiplas detecções                 |
| n_lat_distintas           | UInt32  | Número de latitudes distintas com evento no mês           |
| n_lon_distintas           | UInt32  | Número de longitudes distintas com evento no mês          |


## `df_monthly_biome`
Este conjunto de dados segue a mesma lógica do `df_monthly`, mas com segmentações feitas por bioma.

**Tamanho**: 720 linhas x 8 colunas.

| Feature                 | Tipo        | Descrição                                          |
|-------------------------|-------------|----------------------------------------------------|
| bioma                   | Categorical | Bioma                                              |
| mes_inicio              | Date        | Data correspondente ao primeiro dia do mês         |
| ano_mes_year            | Int32       | Ano da agregação mensal                            |
| ano_mes_month           | Int8        | Mês da agregação mensal (1 – 12)                   |
| n_dias_com_evento       | UInt32      | Número de dias no mês com evento no bioma          |
| total_deteccoes_mes     | UInt32      | Total de detecções no mês para o bioma             |
| media_deteccoes_por_dia | Float64     | Média de detecções por dia no mês                  |
| pct_multiplas_deteccoes | Float64     | Proporção de dias com múltiplas detecções no bioma |


## `climatology`
Este dataset possui dados de médias históricas mensais, agregados ao longo dos anos.

**Tamanho**: 12 linhas x 5 colunas.

| Feature                  | Tipo    | Descrição                                                    |
|--------------------------|---------|--------------------------------------------------------------|
| mes_num                  | Int8    | Mês do ano (1 – 12)                                          |
| mean_n_dias_com_evento   | Float64 | Média histórica de dias com evento no mês                    |
| mean_total_deteccoes_mes | Float64 | Média histórica do total de detecções no mês                 |
| mean_media_deteccoes_dia | Float64 | Média histórica da média diária de detecções                 |
| mean_pct_multiplas       | Float64 | Média histórica da proporção de dias com múltiplas detecções |


## `climatology_biome`
Este conjunto de dados é similar ao `climatology`, mas segmentado por bioma.

**Tamanho**: 72 linhas x 6 colunas.

| Feature                  | Tipo        | Descrição                                                             |
|--------------------------|-------------|-----------------------------------------------------------------------|
| bioma                    | Categorical | Bioma                                                                 |
| mes_num                  | Int8        | Mês do ano (1 – 12)                                                   |
| mean_n_dias_com_evento   | Float64     | Média histórica de dias com evento no mês para o bioma                |
| mean_total_deteccoes_mes | Float64     | Média histórica do total de detecções mensais no bioma                |
| mean_media_deteccoes_dia | Float64     | Média histórica da média diária de detecções no bioma                 |
| mean_pct_multiplas       | Float64     | Média histórica da proporção de dias com múltiplas detecções no bioma |
