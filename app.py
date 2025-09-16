import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.title('Northwind Sales Analysis Dashboard')
st.write('Exploring sales data from the Northwind database to identify patterns and forecast future sales.')

# Load and prepare data
df = pd.read_csv(r"C:\Users\crist\Downloads\Power Bi Edutecno\Northwind Base.csv")

# Convert the VENTAS column to numeric
df['VENTAS'] = df['VENTAS'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Create a dictionary to map Spanish month names to numbers
meses_espanol = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Map the Spanish month names to numbers
df['Mes_num'] = df['Mes'].map(meses_espanol)

# Combine date columns into a single string format
df['Fecha_str'] = df['Año'].astype(str) + '-' + df['Mes_num'].astype(str) + '-' + df['Día'].astype(str)

# Convert the combined string to datetime objects, handling possible errors
df['Fecha'] = pd.to_datetime(df['Fecha_str'], errors='coerce')

# Drop the intermediate columns
df = df.drop(columns=['Mes_num', 'Fecha_str'])


# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    'Sales Overview',
    'Category Analysis',
    'Product Analysis',
    'Customer Analysis',
    'Time Series Analysis',
    'Sales Forecast',
    'Key Findings'
])

with tab1:
    st.header('Sales Overview')
    ventas_por_ano = df.groupby('Año')['VENTAS'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Año', y='VENTAS', data=ventas_por_ano, ax=ax)
    plt.title('Ventas por Año')
    plt.xlabel('Año')
    plt.ylabel('Ventas Totales')
    st.pyplot(fig)
    plt.clf()

    ventas_por_trimestre = df.groupby('Trimestre')['VENTAS'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Trimestre', y='VENTAS', data=ventas_por_trimestre, ax=ax)
    plt.title('Ventas por Trimestre')
    plt.xlabel('Trimestre')
    plt.ylabel('Ventas Totales')
    st.pyplot(fig)
    plt.clf()


with tab2:
    st.header('Category Analysis')
    ventas_por_categoria = df.groupby('CategoryName')['VENTAS'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='CategoryName', y='VENTAS', data=ventas_por_categoria, ax=ax)
    plt.title('Ventas por Categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.clf()

with tab3:
    st.header('Product Analysis')
    ventas_por_producto = df.groupby('ProductName')['VENTAS'].sum().reset_index()
    ventas_por_producto_top10 = ventas_por_producto.sort_values(by='VENTAS', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='ProductName', y='VENTAS', data=ventas_por_producto_top10, ax=ax)
    plt.title('Top 10 Productos por Ventas')
    plt.xlabel('Producto')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.clf()


with tab4:
    st.header('Customer Analysis')
    ventas_por_cliente = df.groupby('Cliente')['VENTAS'].sum().reset_index()
    ventas_por_cliente_top10 = ventas_por_cliente.sort_values(by='VENTAS', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Cliente', y='VENTAS', data=ventas_por_cliente_top10, ax=ax)
    plt.title('Top 10 Clientes por Ventas')
    plt.xlabel('Cliente')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.clf()


with tab5:
    st.header('Time Series Analysis')
    df['AñoMes'] = df['Fecha'].dt.to_period('M')
    ventas_temporal = df.groupby('AñoMes')['VENTAS'].sum().reset_index()
    ventas_temporal['AñoMes'] = ventas_temporal['AñoMes'].astype(str)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='AñoMes', y='VENTAS', data=ventas_temporal, ax=ax)
    plt.title('Tendencia de Ventas a lo Largo del Tiempo (Mensual)')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.clf()

    ventas_por_mes = df.groupby('Mes')['VENTAS'].sum().reindex([
        'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
        'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
    ]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Mes', y='VENTAS', data=ventas_por_mes, ax=ax)
    plt.title('Ventas por Mes (Todos los Años)')
    plt.xlabel('Mes')
    plt.ylabel('Ventas Totales')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.clf()


with tab6:
    st.header('Sales Forecast')
    ventas_mensuales = df.groupby('AñoMes')['VENTAS'].sum().reset_index()
    ventas_mensuales['Periodo'] = range(len(ventas_mensuales))

    X = ventas_mensuales[['Periodo']]
    y = ventas_mensuales['VENTAS']

    # Ensure there are enough samples for splitting
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        ultimo_periodo = ventas_mensuales['Periodo'].max()
        futuros_periodos = pd.DataFrame({'Periodo': range(ultimo_periodo + 1, ultimo_periodo + 7)})
        predicciones = modelo.predict(futuros_periodos)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ventas_mensuales['Periodo'], ventas_mensuales['VENTAS'], 'b-', label='Datos Históricos')
        ax.plot(futuros_periodos['Periodo'], predicciones, 'r--', label='Predicciones')
        ax.set_title('Pronóstico de Ventas - Próximos 6 Meses')
        ax.set_xlabel('Período')
        ax.set_ylabel('Ventas')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.clf()

        st.subheader("Pronóstico de ventas para los próximos 6 meses:")
        for i, pred in enumerate(predicciones, 1):
            st.write(f"Mes {i}: ${pred:,.2f}")
    else:
        st.write("Not enough data to train the forecast model.")


with tab7:
    st.header('Key Findings and Recommendations')

    # Assuming dataframes and variables from previous analysis are available
    # If not, you would need to re-calculate them here or structure the script differently.
    # For this subtask, we assume they are available in the script scope.

    st.subheader('1. Tendencia Temporal')
    # Ensure ventas_por_ano is available
    if 'ventas_por_ano' in locals() and not ventas_por_ano.empty:
        tendencia = 'ascendente' if ventas_por_ano['VENTAS'].iloc[-1] > ventas_por_ano['VENTAS'].iloc[0] else 'descendente'
        mejor_ano = ventas_por_ano.loc[ventas_por_ano['VENTAS'].idxmax()]
        st.write(f"- Las ventas mostraron una tendencia **{tendencia}** durante el período analizado.")
        st.write(f"- El mejor año de ventas fue **{int(mejor_ano['Año'])}** con **${mejor_ano['VENTAS']:,.2f}**.")
    else:
        st.write("- No hay datos suficientes para analizar la tendencia temporal.")


    st.subheader('2. Estacionalidad')
    # Ensure ventas_por_mes is available
    if 'ventas_por_mes' in locals() and not ventas_por_mes.empty:
        mes_max_ventas = ventas_por_mes.loc[ventas_por_mes['VENTAS'].idxmax(), 'Mes']
        st.write(f"- El mes con mayores ventas es: **{mes_max_ventas.capitalize()}**.")
    else:
        st.write("- No hay datos suficientes para analizar la estacionalidad mensual.")


    st.subheader('3. Categorías y Productos Destacados')
    # Ensure ventas_por_categoria and ventas_por_producto are available
    if 'ventas_por_categoria' in locals() and not ventas_por_categoria.empty:
        cat_max_ventas = ventas_por_categoria.loc[ventas_por_categoria['VENTAS'].idxmax()]
        st.write(f"- La categoría con mayores ventas es: **{cat_max_ventas['CategoryName']}** con **${cat_max_ventas['VENTAS']:,.2f}**.")
    else:
         st.write("- No hay datos suficientes para identificar la categoría con mayores ventas.")

    if 'ventas_por_producto' in locals() and not ventas_por_producto.empty:
        prod_max_ventas = ventas_por_producto.loc[ventas_por_producto['VENTAS'].idxmax()]
        st.write(f"- El producto con mayores ventas es: **{prod_max_ventas['ProductName']}** con **${prod_max_ventas['VENTAS']:,.2f}**.")
    else:
        st.write("- No hay datos suficientes para identificar el producto con mayores ventas.")


    st.subheader('4. Clientes Principales')
    # Ensure ventas_por_cliente is available
    if 'ventas_por_cliente' in locals() and not ventas_por_cliente.empty:
        cliente_max_ventas = ventas_por_cliente.loc[ventas_por_cliente['VENTAS'].idxmax()]
        st.write(f"- El cliente con mayores compras es: **{cliente_max_ventas['Cliente']}** con **${cliente_max_ventas['VENTAS']:,.2f}**.")
    else:
        st.write("- No hay datos suficientes para identificar el cliente con mayores compras.")


    st.subheader('5. Pronóstico')
    # Ensure predicciones and ventas_mensuales are available from the Forecast tab
    if 'predicciones' in locals() and 'ventas_mensuales' in locals() and not ventas_mensuales.empty:
        if len(ventas_mensuales) > 1:
            crecimiento_esperado = ((predicciones[-1] - ventas_mensuales['VENTAS'].iloc[-1]) / ventas_mensuales['VENTAS'].iloc[-1]) * 100
            st.write(f"- Se espera un **{'crecimiento' if crecimiento_esperado > 0 else 'decrecimiento'}** del **{abs(crecimiento_esperado):.2f}%** en las ventas para los próximos 6 meses.")
        else:
             st.write("- No hay suficientes datos históricos de ventas para calcular el pronóstico de crecimiento.")
    else:
        st.write("- El pronóstico de ventas no está disponible (asegúrate de que el modelo se haya ejecutado correctamente).")

    st.subheader('Recomendaciones')
    st.markdown("""
    Basado en los hallazgos clave, se recomiendan las siguientes acciones:

    * **Capitalizar la tendencia ascendente:** Continuar invirtiendo en estrategias de marketing y ventas que han demostrado ser efectivas.
    * **Aprovechar la estacionalidad:** Preparar campañas de marketing y ajustar el inventario para el mes de abril, que muestra las mayores ventas.
    * **Enfocarse en productos y categorías de alto rendimiento:** Promocionar activamente "Beverages" y "Côte de Blaye", e investigar los factores de su éxito para replicarlos en otros productos.
    * **Fortalecer la relación con clientes clave:** Ofrecer programas de fidelidad o descuentos especiales a clientes como "QUICK-Stop" para asegurar su continuo negocio.
    * **Monitorear el pronóstico de ventas:** Utilizar el pronóstico para la planificación de inventario, asignación de recursos y establecimiento de objetivos de ventas. Considerar modelos de pronóstico más avanzados si se requiere mayor precisión.
    """)
