from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output


# LEITURA E PREPARAÇÃO DOS DADOS

df = pd.read_csv("ecommerce_estatistica.csv")

# Remover coluna de índice, caso exista
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Converter colunas numéricas
colunas_numericas = [
    "Nota",
    "N_Avaliações",
    "Desconto",
    "Preço",
    "Qtd_Vendidos_Cod"
]

for coluna in colunas_numericas:
    if coluna in df.columns:
        df[coluna] = pd.to_numeric(
            df[coluna].astype(str)
            .str.replace(",", ".", regex=False),
            errors="coerce"
        )

# Verificar as colunas necessárias
colunas_obrigatorias = [
    "Nota",
    "N_Avaliações",
    "Preço",
    "Gênero",
    "Material"
]

colunas_faltando = [
    coluna for coluna in colunas_obrigatorias
    if coluna not in df.columns
]

if colunas_faltando:
    raise ValueError(
        "Colunas não encontradas no CSV: "
        + ", ".join(colunas_faltando)
    )

# Cópia para os gráficos
df = df.copy()

# Remover linhas sem preço para gráficos de preço
df_preco = df.dropna(subset=["Preço"]).copy()

# Dados para dispersão e regressão
df_analise = df.dropna(
    subset=["Preço", "Nota", "N_Avaliações"]
).copy()

# Escala logarítmica exige valores positivos
df_log = df_analise[
    df_analise["N_Avaliações"] > 0
].copy()

# Tema visual dos gráficos
tema = "plotly_white"


# CRIAR OS GRÁFICOS


# Gráfico 1: Histograma dos preços
fig_hist = px.histogram(
    df_preco,
    x="Preço",
    nbins=30,
    title="Distribuição dos preços dos produtos",
    labels={"Preço": "Preço (R$)"},
    template=tema
)

fig_hist.update_layout(
    yaxis_title="Quantidade de produtos"
)


# Gráfico 2: Dispersão entre preço e avaliações
fig_dispersao = px.scatter(
    df_log,
    x="N_Avaliações",
    y="Preço",
    title="Preço por número de avaliações",
    labels={
        "N_Avaliações": "Número de avaliações",
        "Preço": "Preço (R$)"
    },
    template=tema,
    hover_data=["Nota", "Gênero"]
)

fig_dispersao.update_xaxes(type="log")


# Gráfico 3: Mapa de correlação
colunas_corr = [
    coluna for coluna in colunas_numericas
    if coluna in df.columns
]

correlacoes = df[colunas_corr].corr()

fig_corr = px.imshow(
    correlacoes,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlação entre variáveis numéricas",
    template=tema
)


# Gráfico 4: Quantidade de produtos por gênero
dados_genero = (
    df.dropna(subset=["Gênero"])
    .groupby("Gênero")
    .size()
    .reset_index(name="Quantidade")
    .sort_values("Quantidade", ascending=False)
)

fig_genero = px.bar(
    dados_genero,
    x="Gênero",
    y="Quantidade",
    title="Quantidade de produtos por gênero",
    labels={
        "Gênero": "Gênero",
        "Quantidade": "Quantidade de produtos"
    },
    template=tema
)


# Gráfico 5: Distribuição por material
dados_material = (
    df["Material"]
    .fillna("Não informado")
    .value_counts()
    .reset_index()
)

dados_material.columns = ["Material", "Quantidade"]

# Mostrar os 8 materiais mais frequentes e agrupar os demais
if len(dados_material) > 8:
    outros = dados_material.iloc[7:]["Quantidade"].sum()

    dados_material = dados_material.iloc[:7].copy()

    dados_material.loc[len(dados_material)] = [
        "Outros",
        outros
    ]

fig_material = px.pie(
    dados_material,
    names="Material",
    values="Quantidade",
    title="Distribuição dos produtos por material",
    template=tema,
    hole=0.3
)


# Gráfico 6: Densidade dos preços
fig_densidade = px.histogram(
    df_preco,
    x="Preço",
    histnorm="probability density",
    nbins=30,
    title="Distribuição da densidade dos preços",
    labels={"Preço": "Preço (R$)"},
    template=tema
)

fig_densidade.update_layout(
    yaxis_title="Densidade"
)


# Gráfico 7: Regressão entre nota e preço
fig_regressao = px.scatter(
    df_analise,
    x="Nota",
    y="Preço",
    title="Relação entre nota e preço",
    labels={
        "Nota": "Nota do produto",
        "Preço": "Preço (R$)"
    },
    template=tema,
    opacity=0.6
)

# Calcular a linha de regressão, correlação e R²
if len(df_analise) >= 2:
    x = df_analise["Nota"].to_numpy()
    y = df_analise["Preço"].to_numpy()

    coeficientes = np.polyfit(x, y, 1)
    previsoes = np.polyval(coeficientes, x)

    ss_res = np.sum((y - previsoes) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot != 0 else np.nan
    )

    correlacao = df_analise["Nota"].corr(
        df_analise["Preço"]
    )

    x_linha = np.linspace(x.min(), x.max(), 100)
    y_linha = np.polyval(coeficientes, x_linha)

    fig_regressao.add_trace(
        go.Scatter(
            x=x_linha,
            y=y_linha,
            mode="lines",
            name="Regressão linear"
        )
    )

    texto_regressao = (
        f"Correlação: {correlacao:.3f} | "
        f"R²: {r2:.3f}"
    )
else:
    texto_regressao = (
        "Dados insuficientes para calcular a regressão."
    )

fig_regressao.update_layout(
    annotations=[
        dict(
            text=texto_regressao,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False
        )
    ]
)



# CRIAR O DASHBOARD

app = Dash(__name__)

generos = sorted(
    df["Gênero"].dropna().astype(str).unique()
)

opcoes_genero = [
    {"label": "Todos os gêneros", "value": "Todos"}
]

opcoes_genero.extend(
    [
        {"label": genero, "value": genero}
        for genero in generos
    ]
)

app.layout = html.Div(
    [
        html.H1(
            "Dashboard de E-commerce",
            style={"textAlign": "center"}
        ),

        html.P(
            "Análise exploratória dos preços, avaliações "
            "e características dos produtos.",
            style={"textAlign": "center"}
        ),

        html.H2("Filtro interativo"),

        html.Label("Selecione o gênero do produto:"),

        dcc.Dropdown(
            id="filtro-genero",
            options=opcoes_genero,
            value="Todos",
            clearable=False
        ),

        html.Div(
            id="resumo-dados",
            style={
                "padding": "15px",
                "fontWeight": "bold"
            }
        ),

        dcc.Graph(id="grafico-interativo"),

        html.H2("Distribuição dos preços"),
        dcc.Graph(figure=fig_hist),
        dcc.Graph(figure=fig_densidade),

        html.H2("Avaliações e preços"),
        dcc.Graph(figure=fig_dispersao),
        dcc.Graph(figure=fig_regressao),

        html.H2("Características dos produtos"),
        dcc.Graph(figure=fig_genero),
        dcc.Graph(figure=fig_material),

        html.H2("Correlação entre variáveis"),
        dcc.Graph(figure=fig_corr),

        html.H2("Conclusão geral"),

        html.P(
            "O dashboard permite explorar a distribuição dos "
            "preços, comparar avaliações e observar as "
            "características dos produtos. A correlação e a "
            "regressão ajudam a investigar relações entre "
            "as variáveis, mas não demonstram causalidade."
        )
    ],
    style={
        "maxWidth": "1200px",
        "margin": "auto",
        "padding": "20px",
        "fontFamily": "Arial, sans-serif"
    }
)


# CALLBACK DO FILTRO INTERATIVO

@app.callback(
    Output("grafico-interativo", "figure"),
    Output("resumo-dados", "children"),
    Input("filtro-genero", "value")
)
def atualizar_grafico(genero_selecionado):

    if genero_selecionado == "Todos":
        dados = df_log.copy()
    else:
        dados = df_log[
            df_log["Gênero"].astype(str)
            == genero_selecionado
        ].copy()

    fig = px.scatter(
        dados,
        x="N_Avaliações",
        y="Preço",
        hover_data=["Nota", "Gênero", "Material"],
        title=(
            f"Preço por número de avaliações: "
            f"{genero_selecionado}"
        ),
        labels={
            "N_Avaliações": "Número de avaliações",
            "Preço": "Preço (R$)",
            "Nota": "Nota do produto",
            "Gênero": "Gênero",
            "Material": "Material"
        },
        template=tema
    )

    # A escala logarítmica facilita visualizar
    # diferenças entre quantidades de avaliações.
    fig.update_xaxes(type="log")

    resumo = (
        f"Gênero selecionado: {genero_selecionado} | "
        f"Produtos exibidos: {len(dados)}"
    )

    return fig, resumo


# EXECUTAR A APLICAÇÃO

if __name__ == "__main__":
    app.run(debug=True)
