# Dicionário de features

## `df_final_regression`
Este dataset contém os registros de focos de queimadas agrupados por ponto geográfico e dia, isto é, os múltiplos registros por dia originais foram agregados num só. Os critérios de agregação podem ser vistos com mais detalhes no notebook `notebooks/1.0.0-eda-e-tratamento`.

**Tamanho**: 13.586.093 linhas x 21 colunas.

| Feature                  | Tipo        | Descrição                                                                    |
|--------------------------|-------------|------------------------------------------------------------------------------|
| latitude                 | Float64     | Latitude do ponto geográfico do registro                                     |
| longitude                | Float64     | Longitude do ponto geográfico do registro                                    |
| data_dia                 | Date        | Data do evento                                                               |
| bioma                    | Categorical | Bioma associado ao ponto geográfico                                          |
| estado                   | Categorical | Unidade da federação (UF)                                                    |
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
| num_dias_sem_chuva_max   | Float64     | Máx. de dias consecutivos sem chuva até o evento                             |
| precipitacao_max         | Float64     | Máx. de precipitação acumulada até o evento                                  |
| risco_fogo_max           | Float64     | Máx. valor do risco de fogo previsto no dia                                  |
| frp                      | Float64     | Potência Radiativa do Fogo (Fire Radiative Power), em MW                     |

> **Importante**: Algumas colunas de tempo foram tratadas como categóricas, dada a sua natureza cíclica/sazonal.

## `df_final_classification`
Versão diária do `df_final_regression` com foco em classificação da intensidade, incluindo rótulo (baixa, média e alta). Contém as mesmas features do dataset de regressão, exceto FRP, para evitar data leakage.

**Tamanho**: 13.586.093 linhas x 21 colunas.

| Feature           | Tipo        | Descrição                         |
|-------------------|-------------|-----------------------------------|
| label_intensidade | Categorical | Classe de intensidade da queimada |

## `monthly`
Este dataframe faz um resumo geral por mês, contendo métricas agregadas dos registros de incêndios. A tabela a seguir contém a descrição das principais colunas. As demais colunas seguem uma lógica similar.

**Tamanho**: 81 linhas x 25 colunas.

| Feature                      | Tipo    | Descrição                                            |
|------------------------------|---------|------------------------------------------------------|
| dias_no_mes                  | Int8    | Total de dias no mês                                 |
| n_dias_com_evento            | UInt32  | Dias com ao menos uma detecção                       |
| total_deteccoes_mes          | UInt32  | Total de detecções no mês                            |
| deteccoes_por_dia_calendario | Float64 | Média de detecções por dia do calendário             |
| n_pontos_distintos           | UInt32  | Número de pontos geográficos distintos com queimadas |
| frp_total_mes                | Float64 | FRP total acumulado no mês                           |
| frp_por_deteccao             | Float64 | FRP médio por detecção                               |
| mean_dias_sem_chuva_max      | Float64 | Média mensal do máximo de dias sem chuva             |
| mean_precipitacao_max        | Float64 | Média mensal da precipitação máxima                  |
| mean_risco_fogo_max          | Float64 | Média mensal do risco de fogo máximo                 |


## `monthly_biome`
Este conjunto de dados segue a mesma lógica do `monthly`, mas com segmentações feitas por bioma.

**Tamanho**: 500 linhas x 14 colunas.

## `climatology`
Este dataset possui dados de estatísticas históricas mensais, agregados ao longo dos anos.

**Tamanho**: 12 linhas x 4 colunas.

## `climatology_biome`
Este conjunto de dados é similar ao `climatology`, mas segmentado por bioma.

**Tamanho**: 82 linhas x 8 colunas.
