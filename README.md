# Demo Streamlit – Predicción de Devoluciones (Datos Sintéticos)

Este proyecto es una **aplicación de Streamlit** construida como parte de una entrega académica de Ciencia de Datos.  
Simula un caso de negocio donde se desea **predecir la probabilidad de devolución de un pedido** usando un modelo de Machine Learning.

> **Importante:**  
> Todos los datos utilizados en esta app son **sintéticos** (generados artificialmente).  
> No se utiliza ninguna información real de clientes, pedidos o compañía, por temas de **privacidad y confidencialidad**.

---

##  Tecnologías utilizadas

- Python 3.x  
- Streamlit  
- Scikit-learn  
- NumPy  
- Pandas  
- Altair

---

## Modelo de Machine Learning

La aplicación entrena un modelo de **Regresión Logística** sobre datos sintéticos que incluyen variables como:

- `order_value` (valor del pedido)  
- `num_skus` (número de SKUs)  
- `days_since_last_order` (días desde el último pedido)  
- `route_risk` (riesgo logístico simulado)

El modelo predice una probabilidad de devolución y permite ajustar un **umbral de riesgo** para clasificar un pedido como:

- `1` = riesgo de devolución  
- `0` = bajo riesgo

---

##  Funcionalidades de la app

La aplicación de Streamlit incluye:

- **Pantalla de inicio** con descripción del problema.  
- **Visualizaciones**:
  - Histograma de probabilidades de devolución.
  - Barra de tasa de devoluciones por segmento de cliente.
  - Línea de devoluciones simuladas por mes.
- **Dashboard de KPIs**:
  - Accuracy, Precision, Recall, F1-score.
- **Sección de modelo ML**:
  - Entrenamiento de un modelo de Regresión Logística (demo).
  - Simulador de pedido con inputs interactivos.
- **Interactividad**:
  - Slider para el umbral de riesgo.
  - Selectbox para filtrar por segmento.
  - Inputs numéricos y sliders para simular pedidos.

---

## ▶Cómo ejecutar la app localmente

1. Clonar este repositorio o descargar los archivos.
2. Crear y activar un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   venv\Scripts\activate 


pip install -r requirements.txt
pip install streamlit scikit-learn pandas numpy altair
http://localhost:8501



🔒 Nota sobre privacidad
El proyecto original que motivó esta demo utiliza datos reales y un modelo más complejo dentro de un entorno seguro (por ejemplo, Databricks).
Por razones de privacidad, confidencialidad y cumplimiento de políticas internas, ese modelo y esos datos no se exponen aquí.
En su lugar, esta app utiliza un dataset sintético y un modelo sencillo que permiten ilustrar el flujo completo de:

Generación de datos.

Entrenamiento de un modelo.

Visualización de resultados.

Interacción del usuario mediante Streamlit.


👤 Autor

Nombre: Zayd Rogelio Solís Cortés
Curso: [Ciencia de Datos/ Maestría Big Data]

