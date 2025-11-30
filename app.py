import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import altair as alt


st.set_page_config(
    page_title="Modelo de Devoluciones (Demo)",
    layout="wide"
)

st.title("Modelo de Devoluciones - Demo con Datos Sintéticos")

st.markdown("""
Esta aplicación es una **demo académica** que simula un caso de negocio de predicción de devoluciones de pedidos.
Los datos mostrados son **totalmente sintéticos** y el modelo es un ejemplo simple de **Regresión Logística**.
""")

st.sidebar.header("Parámetros de visualización")



@st.cache_data
def generar_datos_sinteticos(n=5000, random_state=42):
    np.random.seed(random_state)

    # Variables "de negocio" simuladas
    order_value = np.random.gamma(shape=5, scale=200, size=n)           
    num_skus = np.random.randint(1, 15, size=n)                         
    days_since_last_order = np.random.randint(0, 60, size=n)           
    route_risk = np.random.uniform(0, 1, size=n)                        
    customer_segment = np.random.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])

    base_logit = (
        -3
        + 0.0005 * order_value
        + 0.1 * num_skus
        - 0.03 * days_since_last_order
        + 2.0 * route_risk
        + np.where(customer_segment == "C", 0.8, 0)
    )

    prob_refusal = 1 / (1 + np.exp(-base_logit))
    y = np.random.binomial(1, prob_refusal)

    df = pd.DataFrame({
        "order_value": order_value,
        "num_skus": num_skus,
        "days_since_last_order": days_since_last_order,
        "route_risk": route_risk,
        "customer_segment": customer_segment,
        "label": y,
        "prob_true": prob_refusal
    })
    return df

df = generar_datos_sinteticos()



X = df[["order_value", "num_skus", "days_since_last_order", "route_risk"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_prob = modelo.predict_proba(X_test)[:, 1]

umbral = st.sidebar.slider(
    "Umbral de riesgo para clasificar devolución",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

y_pred = (y_prob >= umbral).astype(int)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1-score", f"{f1:.3f}")

st.markdown("---")


st.subheader("Visualizaciones del comportamiento del modelo (datos sintéticos)")

st.markdown("**Distribución de probabilidades predichas de devolución (test)**")

hist_df = pd.DataFrame({"probabilidad": y_prob})
bins = st.sidebar.slider("Número de bins del histograma", 5, 50, 20)

hist_chart = alt.Chart(hist_df).mark_bar().encode(
    alt.X("probabilidad", bin=alt.Bin(maxbins=bins)),
    y="count()"
).properties(height=250)

st.altair_chart(hist_chart, use_container_width=True)

st.markdown("**Tasa de devoluciones por segmento de cliente**")

segment_df = df.groupby("customer_segment")["label"].mean().reset_index()
segment_df["tasa_devolucion"] = segment_df["label"]

bar_chart = alt.Chart(segment_df).mark_bar().encode(
    x=alt.X("customer_segment", title="Segmento"),
    y=alt.Y("tasa_devolucion", title="Tasa de devolución"),
    tooltip=["customer_segment", "tasa_devolucion"]
).properties(height=250)

st.altair_chart(bar_chart, use_container_width=True)

st.markdown("**Devoluciones simuladas por mes**")

np.random.seed(42)
df["month"] = np.random.choice(
    ["2025-04", "2025-05", "2025-06", "2025-07"],
    size=len(df)
)

month_df = df.groupby("month")["label"].mean().reset_index()
month_df["tasa_devolucion"] = month_df["label"]

line_chart = alt.Chart(month_df).mark_line(point=True).encode(
    x=alt.X("month", title="Mes"),
    y=alt.Y("tasa_devolucion", title="Tasa de devolución"),
    tooltip=["month", "tasa_devolucion"]
).properties(height=250)

st.altair_chart(line_chart, use_container_width=True)

st.markdown("---")



st.subheader("Explorador de pedidos (datos sintéticos)")

segmento_sel = st.selectbox(
    "Filtrar por segmento de cliente",
    options=["Todos"] + sorted(df["customer_segment"].unique().tolist())
)

if segmento_sel != "Todos":
    df_filtrado = df[df["customer_segment"] == segmento_sel].copy()
else:
    df_filtrado = df.copy()

st.dataframe(df_filtrado.head(50))

st.markdown("---")



st.subheader("Simulador de predicción para un pedido")

st.markdown("""
A continuación puedes introducir los datos de un pedido ficticio y el modelo de **Regresión Logística** 
estimará la probabilidad de que sea devuelto.
""")

c1, c2 = st.columns(2)

with c1:
    input_order_value = st.number_input(
        "Valor del pedido (moneda simulada)",
        min_value=0.0, max_value=10000.0, value=1500.0, step=100.0
    )
    input_num_skus = st.slider(
        "Número de SKUs",
        min_value=1, max_value=30, value=8, step=1
    )

with c2:
    input_days_since = st.slider(
        "Días desde el último pedido",
        min_value=0, max_value=90, value=15, step=1
    )
    input_route_risk = st.slider(
        "Riesgo logístico de la ruta (0–1)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

input_array = np.array([[input_order_value, input_num_skus, input_days_since, input_route_risk]])
prob_simulada = modelo.predict_proba(input_array)[0, 1]
pred_simulada = int(prob_simulada >= umbral)

st.write(f"**Probabilidad estimada de devolución:** `{prob_simulada:.3f}`")
st.write(f"**Clasificación con umbral {umbral:.2f}:** `{pred_simulada}` (1 = riesgo de devolución, 0 = bajo riesgo)")

st.markdown("""
> **Nota:** Todos los datos y resultados son *ficticios* y solo tienen fines demostrativos.
""")

