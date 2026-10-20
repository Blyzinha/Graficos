
# Importa as bibliotecas necessárias
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc

# Lê o arquivo CSV e remove o índice antigo, caso exista
df = pd.read_csv("ecommerce_estatistica.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns="Unnamed: 0")

# Define as colunas numéricas que serão utilizadas
colunas_numericas = [
    "Nota",
    "N_Avaliações",
    "Desconto",
    "Preço",
    "Qtd_Vendidos_Cod"
]

# Converte as colunas para números quando elas existirem
for coluna in colunas_numericas:
    if coluna in df.columns:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

# Cria o aplicativo Dash
app = Dash(__name__)

# Define o título que aparece na aba do navegador
app.title = "Dashboard de E-commerce"


# 1. Histograma: distribuição dos preços
fig_histograma = px.histogram(
    df.dropna(subset=["Preço"]),
    x="Preço",
    nbins=30,
    title="Distribuição dos Preços dos Produtos",
    labels={
        "Preço": "Preço (R$)",
        "count": "Quantidade de produtos"
    }
)

fig_histograma.update_layout(
    xaxis_title="Preço (R$)",
    yaxis_title="Quantidade de produtos"
)


# 2. Dispersão: preço e número de avaliações
df_dispersao = df.dropna(subset=["Preço", "N_Avaliações"])
df_dispersao = df_dispersao[
    (df_dispersao["Preço"] > 0) &
    (df_dispersao["N_Avaliações"] > 0)
]

fig_dispersao = px.scatter(
    df_dispersao,
    x="Preço",
    y="N_Avaliações",
    title="Relação entre Preço e Número de Avaliações",
    labels={
        "Preço": "Preço (R$)",
        "N_Avaliações": "Número de avaliações"
    },
    opacity=0.65
)

# Usa escala logarítmica para facilitar a visualização dos valores
fig_dispersao.update_yaxes(type="log")


# 3. Mapa de calor: correlação entre variáveis numéricas
colunas_disponiveis = [
    coluna for coluna in colunas_numericas
    if coluna in df.columns
]

correlacoes = df[colunas_disponiveis].corr()

fig_correlacao = px.imshow(
    correlacoes,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlação entre as Variáveis Numéricas"
)


# 4. Gráfico de barras: quantidade por gênero
if "Gênero" in df.columns:
    contagem_genero = (
        df["Gênero"]
        .fillna("Não informado")
        .value_counts()
        .head(10)
        .reset_index()
    )

    contagem_genero.columns = ["Gênero", "Quantidade"]

    fig_barras = px.bar(
        contagem_genero,
        x="Quantidade",
        y="Gênero",
        orientation="h",
        title="Os 10 Gêneros Mais Frequentes",
        labels={
            "Gênero": "Gênero",
            "Quantidade": "Quantidade de produtos"
        }
    )

    fig_barras.update_layout(yaxis={"categoryorder": "total ascending"})


# 5. Gráfico de pizza: distribuição por material
if "Material" in df.columns:
    contagem_material = (
        df["Material"]
        .fillna("Não informado")
        .value_counts()
    )

    # Mantém os cinco materiais mais frequentes e agrupa os demais
    materiais_pizza = contagem_material.head(5).copy()

    if len(contagem_material) > 5:
        materiais_pizza["Outros"] = contagem_material.iloc[5:].sum()

    fig_pizza = px.pie(
        names=materiais_pizza.index,
        values=materiais_pizza.values,
        title="Distribuição dos Produtos por Material"
    )


# 6. Gráfico de densidade aproximado usando histograma
fig_densidade = px.histogram(
    df.dropna(subset=["Preço"]),
    x="Preço",
    nbins=40,
    histnorm="probability density",
    marginal="rug",
    title="Distribuição da Densidade dos Preços",
    labels={"Preço": "Preço (R$)"}
)

fig_densidade.update_layout(
    xaxis_title="Preço (R$)",
    yaxis_title="Densidade estimada"
)


# 7. Regressão linear: preço e nota
df_regressao = df.dropna(subset=["Preço", "Nota"])

# Remove valores infinitos e preços negativos
df_regressao = df_regressao[
    (df_regressao["Preço"] >= 0) &
    df_regressao["Preço"].map(lambda x: pd.notna(x) and abs(x) != float("inf")) &
    df_regressao["Nota"].map(lambda x: pd.notna(x) and abs(x) != float("inf"))
]

fig_regressao = px.scatter(
    df_regressao,
    x="Preço",
    y="Nota",
    title="Relação entre Preço e Nota dos Produtos",
    labels={
        "Preço": "Preço (R$)",
        "Nota": "Nota do produto"
    },
    opacity=0.6
)

# Calcula a linha de regressão e o R² quando há dados suficientes
if len(df_regressao) >= 2 and df_regressao["Preço"].nunique() >= 2:
    coeficientes = __import__("numpy").polyfit(
        df_regressao["Preço"],
        df_regressao["Nota"],
        1
    )

    previsoes = (
        coeficientes[0] * df_regressao["Preço"]
        + coeficientes[1]
    )

    # Calcula o coeficiente de determinação R²
    soma_total = (
        (df_regressao["Nota"] - df_regressao["Nota"].mean()) ** 2
    ).sum()

    if soma_total > 0:
        r2 = 1 - (
            (df_regressao["Nota"] - previsoes) ** 2
        ).sum() / soma_total
    else:
        r2 = float("nan")

    fig_regressao.add_trace(
        go.Scatter(
            x=df_regressao["Preço"],
            y=previsoes,
            mode="lines",
            name="Linha de regressão",
            line={"color": "red"}
        )
    )

    # Calcula a correlação entre preço e nota
    correlacao = df_regressao["Preço"].corr(df_regressao["Nota"])

    fig_regressao.update_layout(
        title=(
            "Preço x Nota | "
            f"Correlação: {correlacao:.3f} | "
            f"R²: {r2:.3f}"
        )
    )


# Monta a página com os gráficos e as conclusões
app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "maxWidth": "1200px",
        "margin": "auto",
        "padding": "20px",
        "backgroundColor": "#f5f7fa"
    },
    children=[
        html.H1(
            "Dashboard de Análise de E-commerce",
            style={"textAlign": "center"}
        ),

        html.P(
            "Visualização interativa dos preços, avaliações e "
            "características dos produtos.",
            style={"textAlign": "center"}
        ),

        html.H2("1. Distribuição dos preços"),
        dcc.Graph(figure=fig_histograma),
        html.P(
            "O histograma mostra como os preços estão distribuídos "
            "e em quais faixas há maior concentração de produtos."
        ),

        html.H2("2. Preço e número de avaliações"),
        dcc.Graph(figure=fig_dispersao),
        html.P(
            "A escala logarítmica ajuda a visualizar melhor os produtos "
            "com poucas avaliações, sem esconder os valores mais altos."
        ),

        html.H2("3. Correlação entre variáveis"),
        dcc.Graph(figure=fig_correlacao),
        html.P(
            "A matriz mostra a direção e a intensidade das relações "
            "lineares entre as variáveis numéricas."
        ),

        html.H2("4. Produtos por gênero"),
        dcc.Graph(figure=fig_barras) if "Gênero" in df.columns else
        html.P("A coluna Gênero não foi encontrada no arquivo."),
        html.P(
            "O gráfico destaca os gêneros mais frequentes no conjunto "
            "de dados."
        ),

        html.H2("5. Produtos por material"),
        dcc.Graph(figure=fig_pizza) if "Material" in df.columns else
        html.P("A coluna Material não foi encontrada no arquivo."),
        html.P(
            "A pizza compara a participação dos cinco materiais mais "
            "frequentes com a categoria Outros."
        ),

        html.H2("6. Densidade dos preços"),
        dcc.Graph(figure=fig_densidade),
        html.P(
            "A distribuição ajuda a identificar as faixas de preço "
            "mais comuns e possíveis valores extremos."
        ),

        html.H2("7. Regressão entre preço e nota"),
        dcc.Graph(figure=fig_regressao),
        html.P(
            "A linha de regressão e os valores de correlação e R² "
            "ajudam a avaliar se existe uma relação linear entre preço "
            "e nota. Correlação não significa causalidade."
        ),

        html.H2("Conclusão geral"),
        html.P(
            "O dashboard reúne diferentes perspectivas dos dados do "
            "e-commerce. Os gráficos permitem investigar a distribuição "
            "dos preços, a frequência de categorias, as avaliações e "
            "as relações entre variáveis. As conclusões finais devem "
            "considerar os resultados observados nos gráficos e as "
            "limitações dos dados."
        )
    ]
)


# Inicia o servidor local
if __name__ == "__main__":
    app.run(debug=True)
