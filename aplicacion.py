# =============================================================
# =============================================================
# NIMACLA: herramienta  para el análisis de reglas de asociación
# =============================================================
# =============================================================

MAX_ITEMSETS = 3000
MAX_REGLAS = 50000

import streamlit as st
import pandas as pd
import altair as alt

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
 

from src.procesamiento_datos import (
    codificar_transacciones,
)
from src.mineria import algoritmo_fpgrowth
from src.reglas import (
    generar_reglas,
    construir_evaluaciones_regla,
    filtrar_evaluaciones_regla_por_criterios
)

from src.interpretaciones import (
    InterpretacionAtributoValor,
    InterpretacionAtributoGrupoValores
)


# =============================================================
# FUNCIONES AUXILIARES
# =============================================================

# --------------------------------------------------
# Función: formatear_itemset
# --------------------------------------------------
#   Convierte un itemset en una representación legible en forma de conjunto.
#
# @param itemset Iterable[str]:
#   Conjunto de ítems.
#
# @return str:
#   Cadena de texto en formato "{it1, it2, it3}" con los ítems ordenados.
def formatear_itemset(itemset) -> str:
    return "{" + ", ".join(sorted(itemset)) + "}"



# --------------------------------------------------
# Función: construir_tooltips_grupos
# --------------------------------------------------
#   Construye un diccionario de textos para los ítems
#   generados mediante la interpretación atributo-grupo de valores.
#
#
# @param grupos_por_atributo dict:
#   Diccionario con los grupos definidos por el usuario.
#
# @return dict[str, str]:
#   Diccionario que asocia cada ítem con su descripción.
#
def construir_tooltips_grupos(grupos_por_atributo: dict) -> dict[str, str]: 
    tooltips = {}

    interpretacion = InterpretacionAtributoGrupoValores(grupos_por_atributo)

    for atributo, grupos in grupos_por_atributo.items():
        for definicion_grupo in grupos:
            item_visible = interpretacion._formatear_item(atributo, definicion_grupo)
            definicion = interpretacion._formatear_definicion_grupo(atributo, definicion_grupo)

            tooltips[item_visible] = definicion

    return tooltips

# --------------------------------------------------
# Función: formatear_regla
# --------------------------------------------------
#   Convierte una fila del DataFrame de reglas en una representación legible: X => Y.
#
# @param fila pd.Series:
#   Fila del DataFrame que contiene los campos 'antecedents' y 'consequents'.
#
# @return str:
#   Cadena de texto en formato "{it1, it2} => {it3}".
def formatear_regla(fila) -> str:
    antecedente = formatear_itemset(fila["antecedents"])
    consecuente = formatear_itemset(fila["consequents"])
    return f"{antecedente} => {consecuente}"


# --------------------------------------------------
# Función: construir_tabla_evaluaciones
# --------------------------------------------------
#   Construye un DataFrame a partir de una lista de evaluaciones
#   de regla, mostrando únicamente las medidas indicadas.
#
# @param evaluaciones list[EvaluacionRegla]:
#   Lista de evaluaciones de regla.
#
# @param medidas_visibles list[str]:
#   Medidas que se desean incluir en la tabla.
#
# @return pd.DataFrame:
#   DataFrame listo para su visualización en la interfaz.
def construir_tabla_evaluaciones(evaluaciones, medidas_visibles) -> pd.DataFrame:
    filas = []

    for evaluacion in evaluaciones:
        fila = {"regla": str(evaluacion.regla)}

        for medida in medidas_visibles:
            valor = evaluacion.medidas.get(medida)

            if valor == float("inf"):
                fila[medida] = "inf"
            else:
                fila[medida] = valor

        filas.append(fila)

    return pd.DataFrame(filas)

# --------------------------------------------------
# Función: es_atributo_numerico
# --------------------------------------------------
#   Determina si una columna del dataset es numérica.
#
# @param serie pd.Series:
#   Columna del DataFrame que se desea analizar.
#
# @return bool:
#   True si todos los valores no nulos son numéricos,
#   False en caso contrario.
#
def es_atributo_numerico(serie: pd.Series) -> bool:
    serie_convertida = pd.to_numeric(serie.dropna(), errors="coerce")
    return serie_convertida.notna().all()


# --------------------------------------------------
# Función: obtener_firma_datos
# --------------------------------------------------
#   Construye una firma sencilla del conjunto de datos cargado.
#   Se utiliza para detectar cambios de dataset entre ejecuciones
#   de la aplicación y reiniciar los resultados dependientes.
#
# @param datos pd.DataFrame:
#   DataFrame tabular cargado por el usuario.
#
# @param nombre_dataset str:
#   Nombre del fichero utilizado como origen de los datos.
#
# @return tuple:
#   Firma asociada al conjunto de datos.
#
def obtener_firma_datos(datos: pd.DataFrame, nombre_dataset: str) -> tuple:
    return (
        nombre_dataset,
        datos.shape,
        tuple(datos.columns),
        int(pd.util.hash_pandas_object(datos, index=True).sum()),
    )


# --------------------------------------------------
# Función: limpiar_estado_dependiente
# --------------------------------------------------
#   Elimina del estado de sesión la información que depende
#   del dataset cargado o de la interpretación seleccionada.
#
#   Esta función evita que se reutilicen itemsets, reglas,
#   evaluaciones, filtros o grupos definidos para una configuración
#   anterior.
#
def limpiar_estado_dependiente() -> None:
    claves_a_eliminar = [
        "items",
        "transacciones",
        "datos_codificados",
        "itemsets_frecuentes",
        "reglas_df",
        "evaluaciones",
        "reglas_filtradas",
        "medidas_filtrado",
        "criterios_filtrado",
        "grupos_por_atributo",
        "tooltips_grupos",
        "num_filas_datosoriginales",
        "num_filas_datoscodificados",
        "num_filas_itemsets",
        "num_filas_reglas",
        "num_filas_evaluaciones",
        "num_filas_filtradas",
        "num_reglas_informe",
    ]

    for clave in claves_a_eliminar:
        st.session_state.pop(clave, None)

    for clave in [
        "bloquear_soporte",
        "bloquear_generacion_reglas",
        "bloquear_evaluacion",
        "bloquear_filtrado",
    ]:
        st.session_state[clave] = False

    st.session_state["mostrar_datos_originales"] = False
    st.session_state["mostrar_datos_codificados"] = False

    prefijos_configuracion_grupos = (
        "atributos_agrupados",
        "num_intervalos_",
        "nombre_grupo_",
        "tipo_izq_",
        "minimo_",
        "maximo_",
        "tipo_der_",
        "num_grupos_",
        "valores_grupo_",
    )

    for clave in list(st.session_state.keys()):
        if clave.startswith(prefijos_configuracion_grupos):
            st.session_state.pop(clave, None)

# --------------------------------------------------
# Función: configurar_interpretacion_atributo_valor
# --------------------------------------------------
#   Configura la interpretación atributo-valor.
#
# @param datos pd.DataFrame:
#   DataFrame tabular cargado por el usuario.
#
# @return interpretacion: instancia de la interpretación atributo-valor.
#
def configurar_interpretacion_atributo_valor(_datos: pd.DataFrame):
    return InterpretacionAtributoValor()

# --------------------------------------------------
# Función: configurar_grupos_atributo
# --------------------------------------------------
#   Construye la configuración de grupos asociada a un atributo.
#
#   Se distingue entre atributos numéricos y categóricos:
#
#       - En atributos numéricos, permite definir intervalos
#         abiertos o cerrados.
#
#       - En atributos categóricos, permite definir grupos
#         mediante subconjuntos de valores.
#
# @param datos pd.DataFrame:
#   DataFrame cargado por el usuario.
#
# @param atributo str:
#   Nombre del atributo cuyos grupos se desean configurar.
#
# @return list[dict]:
#   Lista de diccionarios que representan los grupos definidos
#   para el atributo.
#
def configurar_grupos_atributo(datos: pd.DataFrame, atributo: str) -> list[dict]:
    grupos = []

    if es_atributo_numerico(datos[atributo]):
        datos[atributo] = pd.to_numeric(datos[atributo], errors="coerce")

        num_intervalos = st.number_input(
            f"Número de intervalos para {atributo}",
            min_value=1,
            value=1,
            step=1,
            key=f"num_intervalos_{atributo}"
        )

        for i in range(num_intervalos):
            nombre_por_defecto = f"Grupo {i + 1}"
            nombre_grupo = st.text_input(
                "Nombre del grupo",
                value=nombre_por_defecto,
                key=f"nombre_grupo_{atributo}_{i}"
            )

            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

            with col1:
                tipo_izq = st.selectbox(
                    "Izquierda",
                    options=["[", "("],
                    key=f"tipo_izq_{atributo}_{i}"
                )

            with col2:
                minimo = st.number_input(
                    "Mínimo",
                    value=0.0,
                    key=f"minimo_{atributo}_{i}"
                )

            with col3:
                maximo = st.number_input(
                    "Máximo",
                    value=1.0,
                    key=f"maximo_{atributo}_{i}"
                )

            with col4:
                tipo_der = st.selectbox(
                    "Derecha",
                    options=["]", ")"],
                    key=f"tipo_der_{atributo}_{i}"
                )

            grupos.append({
                "nombre": nombre_grupo,
                "tipo": "intervalo",
                "minimo": minimo,
                "maximo": maximo,
                "cerrado_izquierda": tipo_izq == "[",
                "cerrado_derecha": tipo_der == "]",
            })

    else:
        valores_posibles = sorted(
            datos[atributo].dropna().astype(str).unique()
        )

        num_grupos = st.number_input(
            f"Número de grupos para {atributo}",
            min_value=1,
            value=1,
            step=1,
            key=f"num_grupos_{atributo}"
        )

        columnas = st.columns(min(num_grupos, 3))

        for i in range(num_grupos):
            with columnas[i % len(columnas)]:
                nombre_por_defecto = f"Grupo {i + 1}"

                nombre_grupo = st.text_input(
                    "Nombre del grupo",
                    value=nombre_por_defecto,
                    key=f"nombre_grupo_{atributo}_{i}"
                )

                valores_grupo = st.multiselect(
                    "Valores del grupo",
                    options=valores_posibles,
                    default=valores_posibles,
                    key=f"valores_grupo_{atributo}_{i}"
                )

                grupos.append({
                    "nombre": nombre_grupo,
                    "tipo": "categorico",
                    "valores": set(valores_grupo)
                })

    return grupos


# --------------------------------------------------
# Función: configurar_interpretacion_atributo_grupo_valores
# --------------------------------------------------
#   Configura la interpretación atributo-grupo de valores.
#
#   La función permite seleccionar los atributos sobre los que
#   se desean definir grupos de valores.
#   Una vez definidos los grupos, se construye la instancia
#   correspondiente y se devuelve.
#
# @param datos pd.DataFrame:
#   DataFrame tabular cargado por el usuario.
#
# @return interpretacion: instancia de la interpretación atributo-grupo de valores.
#
def configurar_interpretacion_atributo_grupo_valores(datos: pd.DataFrame):
    st.markdown("#### Definición de grupos de valores")

    atributos_agrupados = st.multiselect(
        "Selecciona los atributos para los que quieres definir grupos",
        options=list(datos.columns),
        key="atributos_agrupados"
    )

    grupos_guardados = st.session_state.get("grupos_por_atributo")

    if (
        grupos_guardados is not None
        and set(grupos_guardados.keys()) != set(atributos_agrupados)
    ):
        st.session_state.pop("grupos_por_atributo", None)
        st.session_state.pop("tooltips_grupos", None)

    grupos_por_atributo = {}

    for atributo in atributos_agrupados:
        st.markdown(f"##### {atributo}")
        grupos_por_atributo[atributo] = configurar_grupos_atributo(datos, atributo)

    if st.button("Guardar grupos", use_container_width=True):
        if grupos_por_atributo:
            st.session_state["grupos_por_atributo"] = grupos_por_atributo
            st.success("Configuración de grupos guardada.")
        else:
            st.warning("Define al menos un grupo para aplicar esta interpretación.")

    if "grupos_por_atributo" not in st.session_state:
        st.stop()

    interpretacion = InterpretacionAtributoGrupoValores(
        st.session_state["grupos_por_atributo"]
    )

    return interpretacion

# --------------------------------------------------
# Función: construir_tabla_analisis
# --------------------------------------------------
#   Construye una tabla con información
#   para el resumen analítico de reglas.
#
#   La tabla incluye la representación de cada regla,
#   su antecedente, su consecuente, su longitud, un identificador
#   interno y los valores de las medidas de interés calculadas.
#
# @param evaluaciones list[EvaluacionRegla]:
#   Lista de evaluaciones de regla sobre la que se construye el análisis.
#
# @return pd.DataFrame:
#   DataFrame utilizado para el resumen descriptivo, las visualizaciones
#   y la generación del informe.
#

def construir_tabla_analisis(evaluaciones) -> pd.DataFrame: 
    filas = []

    for i, evaluacion in enumerate(evaluaciones):
        fila = {
            "id_regla": i,
            "regla": str(evaluacion.regla),
            "antecedente": formatear_itemset(evaluacion.regla.antecedente),
            "consecuente": formatear_itemset(evaluacion.regla.consecuente),
            "longitud_regla": len(evaluacion.regla.antecedente) + len(evaluacion.regla.consecuente),
        }

        for medida, valor in evaluacion.medidas.items():
            fila[medida] = valor

        filas.append(fila)

    return pd.DataFrame(filas)

# --------------------------------------------------
# Función: generar_informe_pdf
# --------------------------------------------------
#   Genera un informe PDF con la configuración del análisis,
#   el resumen descriptivo y una selección de reglas.
#
# @param tabla_analisis pd.DataFrame:
#   Tabla auxiliar utilizada en el resumen analítico.
#
# @param resumen dict:
#   Diccionario con los indicadores del resumen descriptivo.
#
# @param origen_analisis str:
#   Conjunto de reglas sobre el que se ha construido el análisis.
#
# @param max_reglas int:
#   Número máximo de reglas que se incluirán en el informe.
#
# @return BytesIO:
#   Buffer en memoria con el contenido del PDF.
def generar_informe_pdf(
    tabla_analisis: pd.DataFrame,
    resumen: dict,
    origen_analisis: str,
    max_reglas: int = 300
) -> BytesIO:

    buffer = BytesIO()

    documento = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    estilos = getSampleStyleSheet()

    estilo_titulo = ParagraphStyle(
        "TituloInforme",
        parent=estilos["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=12
    )

    estilo_subtitulo = ParagraphStyle(
        "SubtituloInforme",
        parent=estilos["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6
    )

    estilo_normal = ParagraphStyle(
        "TextoInforme",
        parent=estilos["Normal"],
        fontSize=8,
        leading=10
    )

    estilo_tabla = ParagraphStyle(
        "TextoTabla",
        parent=estilos["Normal"],
        fontSize=6,
        leading=7
    )

    elementos = []

    # =====================================================
    # Título
    # =====================================================

    elementos.append(
        Paragraph(
            "NIMACLA: herramienta de análisis de reglas de asociación",
            estilo_titulo
        )
    )

    elementos.append(
        Paragraph(
            "Informe de análisis",
            estilos["Heading2"]
        )
    )

    fecha_generacion = datetime.now().strftime("%d/%m/%Y %H:%M")

    elementos.append(
        Paragraph(
            f"Fecha de generación: {fecha_generacion}",
            estilo_normal
        )
    )

    elementos.append(
        Paragraph(
            f"Resumen calculado sobre el conjunto de {origen_analisis}.",
            estilo_normal
        )
    )

    elementos.append(Spacer(1, 0.4 * cm))

    # =====================================================
    # Configuración del análisis
    # =====================================================

    elementos.append(
        Paragraph("Configuración del análisis", estilo_subtitulo)
    )

    criterios_filtrado = st.session_state.get("criterios_filtrado", {})

    if criterios_filtrado:
        filtros_texto = "<br/>".join(
            [
                f"{medida} ≥ {umbral:.3f}"
                for medida, umbral in criterios_filtrado.items()
            ]
        )
    else:
        filtros_texto = "No se han aplicado filtros."

    medidas_evaluacion = st.session_state.get("medidas_evaluacion", [])

    if medidas_evaluacion:
        medidas_texto = ", ".join(medidas_evaluacion)
    else:
        medidas_texto = "-"

    configuracion = [
        ["Parámetro", "Valor"],
        ["Dataset", st.session_state.get("nombre_dataset", "-")],
        ["Interpretación tabular", st.session_state.get("tipo_interpretacion", "-")],
        ["Tiempo máximo FP-Growth (s)", st.session_state.get("tiempo_maximo", "-")],
    ]

    tabla_configuracion = Table(
        [
            [
                Paragraph(str(celda), estilo_normal)
                for celda in fila
            ]
            for fila in configuracion
        ],
        colWidths=[5 * cm, 19 * cm]
    )

    tabla_configuracion.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    elementos.append(tabla_configuracion)
    elementos.append(Spacer(1, 0.4 * cm))

    # =====================================================
    # Resumen descriptivo
    # =====================================================

    elementos.append(
        Paragraph("Resumen descriptivo", estilo_subtitulo)
    )

    tabla_resumen = Table(
        [
            [
                Paragraph(str(indicador), estilo_normal),
                Paragraph(str(valor), estilo_normal)
            ]
            for indicador, valor in resumen.items()
        ],
        colWidths=[7 * cm, 17 * cm]
    )

    tabla_resumen.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    elementos.append(tabla_resumen)

    elementos.append(PageBreak())

    # =====================================================
    # Tabla de reglas
    # =====================================================

    elementos.append(
        Paragraph("Reglas incluidas en el informe", estilo_subtitulo)
    )

    elementos.append(
        Paragraph(
            "Por razones de legibilidad, el informe muestra únicamente las primeras reglas "
            "del conjunto analizado. La selección se limita para evitar generar un documento "
            "excesivamente extenso.",
            estilo_normal
        )
    )

    elementos.append(Spacer(1, 0.3 * cm))

    columnas_base = ["regla"]

    columnas_medidas = [
        medida
        for medida in opciones_medidas.values()
        if medida in tabla_analisis.columns
    ]

    columnas_adicionales = [
        columna
        for columna in ["tipo_dependencia"]
        if columna in tabla_analisis.columns
    ]

    columnas_reglas = columnas_base + columnas_medidas + columnas_adicionales

    nombres_columnas = {
        valor: clave
        for clave, valor in opciones_medidas.items()
    }

    nombres_columnas["regla"] = "Regla"
    nombres_columnas["tipo_dependencia"] = "Tipo de dependencia"

    tabla_reglas_pdf = tabla_analisis[columnas_reglas].head(max_reglas).copy()

    for columna in tabla_reglas_pdf.columns:
        if columna != "regla" and columna != "tipo_dependencia":
            tabla_reglas_pdf[columna] = tabla_reglas_pdf[columna].apply(
                lambda x: "∞" if x == float("inf") else (
                    f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
                )
            )

    datos_tabla = [[
        nombres_columnas.get(columna, columna)
        for columna in columnas_reglas
    ]]

    for _, fila in tabla_reglas_pdf.iterrows():
        datos_tabla.append([
            Paragraph(str(fila[columna]), estilo_tabla)
            for columna in columnas_reglas
        ])

    anchos_columnas = []

    for columna in columnas_reglas:
        if columna == "regla":
            anchos_columnas.append(10 * cm)
        elif columna == "tipo_dependencia":
            anchos_columnas.append(3.4 * cm)
        else:
            anchos_columnas.append(2.1 * cm)

    tabla_reglas = Table(
        datos_tabla,
        colWidths=anchos_columnas,
        repeatRows=1
    )

    tabla_reglas.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    elementos.append(tabla_reglas)

    if len(tabla_analisis) > max_reglas:
        elementos.append(Spacer(1, 0.3 * cm))
        elementos.append(
            Paragraph(
                f"Se muestran {max_reglas} reglas de un total de {len(tabla_analisis)} reglas analizadas.",
                estilo_normal
            )
        )

    documento.build(elementos)

    buffer.seek(0)
    return buffer


#DICCIONARIOS
CONFIGURADORES_INTERPRETACION = {
    "Atributo-valor": configurar_interpretacion_atributo_valor,
    "Atributo-grupo de valores": configurar_interpretacion_atributo_grupo_valores,
}

opciones_medidas = {
    "Soporte": "soporte",
    "Confianza": "confianza",
    "Lift": "lift",
    "Leverage": "leverage",
    "Conviction": "conviction",
    "Factor de certeza": "factor_certeza",
    "Información mutua": "informacion_mutua",
}
opciones_medidas_generacion = {
    "Confianza": "confidence",
    "Lift": "lift",
    "Leverage": "leverage",
    "Conviction": "conviction",
}

# =============================================================
# CONFIGURACIÓN INICIAL
# =============================================================
st.set_page_config(page_title="NIMACLA", layout="wide")


# =============================================================
# CABECERA CON EL TÍTULO DE LA HERRAMIENTA
# =============================================================

st.markdown("""
<style>

/* -------------------------------------------------
   Oculta parcialmente el header por defecto
------------------------------------------------- */
header[data-testid="stHeader"] {
    background: transparent;
}

/* -------------------------------------------------
   Barra superior fija
------------------------------------------------- */
.barra-superior {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;

    height: 70px;

    background-color: #1E3A5F;

    display: flex;
    align-items: center;
    justify-content: center;

    z-index: 999999;

    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

/* -------------------------------------------------
   Texto del título
------------------------------------------------- */
.titulo-barra {
    color: white;

    font-size: 25px;
    font-weight: 600;

    letter-spacing: 0.5px;

    font-family: "Source Sans Pro", sans-serif;
}

/* -------------------------------------------------
   Margen superior del contenido principal
------------------------------------------------- */
.block-container {
    padding-top: 95px;
}
            

</style>

<div class="barra-superior">
    <div class="titulo-barra">
        NIMACLA: Herramienta de análisis de reglas de asociación
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# ESTADO INICIAL
# =============================================================

if "mostrar_datos_originales" not in st.session_state:
    st.session_state["mostrar_datos_originales"] = False

if "mostrar_datos_codificados" not in st.session_state:
    st.session_state["mostrar_datos_codificados"] = False


# =============================================================
# 1. CARGA DE DATOS
# =============================================================

with st.expander("Carga de datos", expanded=True):

    archivo_subido = st.file_uploader("Adjuntar archivo CSV", type=["csv"])
    usar_ejemplo = st.checkbox("Usar el CSV de ejemplo", value=False)

    datos = None
    nombre_dataset_actual = None

    if usar_ejemplo:
        datos = pd.read_csv("data/articulos_ropa_atributo_valor.csv")
        nombre_dataset_actual = "articulos_ropa_atributo_valor.csv"
    elif archivo_subido is not None:
        datos = pd.read_csv(archivo_subido)
        nombre_dataset_actual = archivo_subido.name
    
    tipo_interpretacion = st.selectbox(
        "Interpretación tabular",
        options=[
            "Atributo-valor",
            "Atributo-grupo de valores"
        ]
    )

    if datos is not None:
        firma_datos = obtener_firma_datos(datos, nombre_dataset_actual)
        datos_cambiados = st.session_state.get("firma_datos") != firma_datos
        interpretacion_cambiada = (
            st.session_state.get("tipo_interpretacion") != tipo_interpretacion
        )

        if datos_cambiados or interpretacion_cambiada:
            limpiar_estado_dependiente()

        st.session_state["datos"] = datos
        st.session_state["nombre_dataset"] = nombre_dataset_actual
        st.session_state["firma_datos"] = firma_datos
        st.session_state["tipo_interpretacion"] = tipo_interpretacion
        

        if {"tid", "item"}.issubset(datos.columns):
            st.error(
                "El archivo cargado tiene formato transaccional con columnas 'tid' e 'item'. "
                "La herramienta espera un CSV tabular, con los atributos en la cabecera "
                "y las tuplas en las filas."
            )
            st.stop()

        configurador = CONFIGURADORES_INTERPRETACION[tipo_interpretacion]
        interpretacion = configurador(datos)

        if tipo_interpretacion == "Atributo-grupo de valores":
            st.session_state["tooltips_grupos"] = construir_tooltips_grupos(
                st.session_state["grupos_por_atributo"]
            )
        else:
            st.session_state["tooltips_grupos"] = {}

        items = interpretacion.construir_items(datos)  
        transacciones = interpretacion.construir_transacciones(datos)  

        datos_codificados = codificar_transacciones(transacciones)

        st.session_state["items"] = items
        st.session_state["transacciones"] = transacciones
        st.session_state["datos_codificados"] = datos_codificados

        st.write(f"El dataset tabular se ha interpretado mediante {tipo_interpretacion}.")

    elif "datos" in st.session_state:
        limpiar_estado_dependiente()

        for clave in [
            "datos",
            "nombre_dataset",
            "firma_datos",
            "tipo_interpretacion",
        ]:
            st.session_state.pop(clave, None)

    if "datos" in st.session_state:
        if st.button("Mostrar datos originales"):
            st.session_state["mostrar_datos_originales"] = not st.session_state["mostrar_datos_originales"]

        if st.session_state["mostrar_datos_originales"]:
            st.markdown("#### Datos originales")
            col_input, col_vacio = st.columns([1, 3])

            with col_input:
                num_filas = st.number_input(
                    "Número de filas a mostrar",
                    min_value=1,
                    max_value=len(st.session_state["datos"]),
                    value=min(25, len(st.session_state["datos"])),
                    step=1,
                    key="num_filas_datosoriginales"
                )
            st.dataframe(st.session_state["datos"].head(num_filas), use_container_width=True)

    if "datos_codificados" in st.session_state:
        if st.button("Mostrar base de datos transaccional"):
            st.session_state["mostrar_datos_codificados"] = not st.session_state["mostrar_datos_codificados"]

        if st.session_state["mostrar_datos_codificados"]:
            st.markdown("#### Base de datos transaccional")
            col_input, col_vacio = st.columns([1, 3])

            with col_input:
                num_filas = st.number_input(
                    "Número de filas a mostrar",
                    min_value=1,
                    max_value=len(st.session_state["datos_codificados"]),
                    value=min(25, len(st.session_state["datos_codificados"])),
                    step=1,
                    key="num_filas_datoscodificados"
                )

            datos_mostrar = st.session_state["datos_codificados"].head(num_filas) 

            tooltips_grupos = st.session_state.get("tooltips_grupos", {})

            column_config = {
                columna: st.column_config.CheckboxColumn(
                    help=tooltips_grupos[columna]
                )
                for columna in datos_mostrar.columns
                if columna in tooltips_grupos
            }

            st.dataframe(
                datos_mostrar,
                use_container_width=True,
                column_config=column_config
            )


# =============================================================
# 2. ANÁLISIS GENERAL
# =============================================================

if "datos_codificados" in st.session_state:

    st.markdown("### Análisis de reglas de asociación")
    st.caption("Generación, evaluación y filtrado de reglas de asociación.")

    # ---------------------------------
    # ITEMSETS FRECUENTES
    # ---------------------------------
    with st.expander("1. Itemsets frecuentes", expanded=True):
        if "transacciones" in st.session_state:
            st.caption("Extracción de itemsets frecuentes a partir del soporte mínimo.")

            # -------------------------
            # CONFIGURACIÓN EXTRACCIÓN
            # -------------------------
            st.markdown("#### Configuración")

            col_input, col_vacio = st.columns([1, 3])

            with col_input:

                if "bloquear_soporte" not in st.session_state:
                    st.session_state["bloquear_soporte"] = False
                soporte_minimo = st.number_input(
                    "Soporte mínimo",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.1,
                    step=0.001,
                    format="%.3f",
                    key="soporte_minimo",
                    disabled=st.session_state["bloquear_soporte"]
                )
                tiempo_maximo = st.number_input(
                    "Tiempo máximo de ejecución (segundos)",
                    min_value=10,
                    max_value=36000,
                    value=300,
                    step=10,
                    key="tiempo_maximo",
                    disabled=st.session_state["bloquear_soporte"]
                )

            # Botón de desbloqueo
            if (
                st.session_state["bloquear_soporte"]
                and "itemsets_frecuentes" in st.session_state
            ):

                st.info("Soporte mínimo bloqueado.")
                if st.button("Modificar soporte mínimo o tiempo máximo"):
                    st.session_state["bloquear_soporte"] = False
                    st.session_state["bloquear_generacion_reglas"] = False
                    st.session_state["bloquear_evaluacion"] = False
                    st.session_state["bloquear_filtrado"] = False
                    for clave in [
                        "itemsets_frecuentes",
                        "reglas_df",
                        "evaluaciones",
                        "reglas_filtradas",
                    ]:
                        st.session_state.pop(clave, None)

                    st.rerun()

            st.markdown("---")
            st.markdown("#### Resultados")

            # Botón de generación
            if st.button("Generar itemsets frecuentes", use_container_width=True):

                try:
                    with st.spinner(
                        "Generando itemsets frecuentes. Este proceso puede tardar unos minutos..."
                    ):
                        itemsets_frecuentes = algoritmo_fpgrowth(
                            st.session_state["datos_codificados"],
                            soporte_minimo=soporte_minimo,
                            tiempo_maximo=tiempo_maximo
                        )

                except MemoryError:
                    st.error(
                        "No hay memoria suficiente para completar la extracción de itemsets frecuentes. "
                        " Prueba con un valor más alto de soporte mínimo."
                    )
                    st.stop()
                
                except TimeoutError:
                    st.error(
                        "La extracción de itemsets ha superado el tiempo máximo permitido."
                        " Prueba con un valor más alto de soporte mínimo."
                    )
                    st.stop()

                except Exception as error:
                    st.error(
                        f"Se ha producido un error durante la extracción de itemsets: {error}"
                    )
                    st.stop()

                if len(itemsets_frecuentes) > MAX_ITEMSETS:
                    st.error(
                        f"Se han generado {len(itemsets_frecuentes)} itemsets frecuentes, "
                        f"superando el límite establecido de {MAX_ITEMSETS}. "
                        " Prueba con un valor más alto de soporte mínimo."
                    )
                    st.stop()

                st.session_state["itemsets_frecuentes"] = itemsets_frecuentes
                st.session_state["bloquear_soporte"] = True

                for clave in [
                    "reglas_df",
                    "evaluaciones",
                    "reglas_filtradas",
                ]:
                    st.session_state.pop(clave, None)

                st.rerun()


            if "itemsets_frecuentes" in st.session_state:
                itemsets_visibles = st.session_state["itemsets_frecuentes"].copy()

                st.write(f"Número de itemsets frecuentes: {len(itemsets_visibles)}")
                st.write(f"Soporte máximo: {itemsets_visibles['support'].max():.4f}")
                st.write(f"Soporte mínimo: {itemsets_visibles['support'].min():.4f}")

                with st.expander("Ver itemsets frecuentes", expanded=False):

                    itemsets_visibles["itemsets"] = itemsets_visibles["itemsets"].apply(formatear_itemset)
                    
                    if len(itemsets_visibles) == 0:
                        st.warning("No se han generado itemsets frecuentes con el soporte mínimo seleccionado.")
                    else:
                        col_input, col_vacio = st.columns([1, 3])

                        with col_input:
                            num_filas = st.number_input(
                                "Número de filas a mostrar",
                                min_value=1,
                                max_value=len(itemsets_visibles),
                                value=min(25, len(itemsets_visibles)),
                                step=1,
                                key="num_filas_itemsets"
                            )
                        tabla_itemsets = (
                            itemsets_visibles[["itemsets", "support"]]
                            .head(num_filas)
                            .rename(columns={"support": "soporte"})
                        )
                        st.dataframe(tabla_itemsets, use_container_width=True)
                        

                    
    # ---------------------------------
    # REGLAS GENERADAS
    # ---------------------------------
    with st.expander("2. Reglas generadas", expanded=False):
        if "itemsets_frecuentes" in st.session_state:
            st.caption("Generación de reglas a partir de los itemsets frecuentes.")

            # -------------------------
            # CONFIGURACIÓN GENERACIÓN DE REGLAS
            # -------------------------
            st.markdown("#### Configuración")

            col_input, col_vacio = st.columns([1, 3])

            with col_input:

                if "bloquear_generacion_reglas" not in st.session_state:
                    st.session_state["bloquear_generacion_reglas"] = False

                etiqueta_medida_generacion = st.selectbox(
                    "Selecciona una medida para generar reglas",
                    options=[
                        "Confianza",
                        "Lift",
                        "Leverage",
                        "Conviction",
                    ],
                    key="medida_generacion",
                    disabled=st.session_state["bloquear_generacion_reglas"]
                )

                medida_generacion = opciones_medidas_generacion[etiqueta_medida_generacion]

                umbral_generacion = st.number_input(
                    "Umbral de la medida escogida",
                    value=0.0,
                    step=0.001,
                    format="%.3f",
                    key="umbral_generacion",
                    disabled=st.session_state["bloquear_generacion_reglas"]
                )
            
            if (
                st.session_state["bloquear_generacion_reglas"]
                and "reglas_df" in st.session_state
            ):
                st.info("Configuración de generación bloqueada.")

                if st.button("Modificar configuración de generación"):
                    st.session_state["bloquear_generacion_reglas"] = False
                    st.session_state["bloquear_evaluacion"] = False
                    st.session_state["bloquear_filtrado"] = False

                    for clave in [
                        "reglas_df",
                        "evaluaciones",
                        "reglas_filtradas",
                    ]:
                        st.session_state.pop(clave, None)

                    st.rerun()

            st.markdown("---")
            st.markdown("#### Resultados")

            if st.button("Generar reglas", use_container_width=True):
                try:
                    with st.spinner("Generando reglas de asociación. Este proceso puede tardar unos minutos..."):
                        reglas_df = generar_reglas(
                            st.session_state["itemsets_frecuentes"],
                            medida=medida_generacion,
                            umbral_minimo=umbral_generacion
                        )

                except MemoryError:
                    st.error(
                        "No hay memoria suficiente para generar las reglas. "
                        " Prueba con un umbral más restrictivo."
                    )
                    st.stop()

                except Exception as error:
                    st.error(
                        f"Se ha producido un error durante la generación de reglas: {error}"
                    )
                    st.stop()

                if len(reglas_df) > MAX_REGLAS:
                    st.error(
                        f"Se han generado {len(reglas_df)} reglas, "
                        f"superando el límite establecido de {MAX_REGLAS}. "
                        " Prueba con un umbral más restrictivo."
                    )
                    st.stop()

                st.session_state["reglas_df"] = reglas_df
                st.session_state["bloquear_generacion_reglas"] = True

                for clave in [
                    "evaluaciones",
                    "reglas_filtradas",
                ]:
                    st.session_state.pop(clave, None)

                st.rerun()

            if "reglas_df" in st.session_state:
                reglas_visibles = st.session_state["reglas_df"].copy()

                if reglas_visibles.empty:
                    st.warning(
                        "No se han generado reglas con la medida base y el umbral seleccionados."
                    )
                else:
                    st.write(f"Número de reglas generadas: {len(reglas_visibles)}")
                    st.write(f"Medida base: {etiqueta_medida_generacion}")
                    st.write(f"Umbral aplicado: {umbral_generacion}")

                    with st.expander("Ver reglas generadas", expanded=False):

                        reglas_visibles["regla"] = reglas_visibles.apply(formatear_regla, axis=1)

                        columnas_a_mostrar = ["regla", "support"]

                        if medida_generacion != "support":
                            columnas_a_mostrar.append(medida_generacion)

                        if len(reglas_visibles[columnas_a_mostrar]) == 0:
                            st.warning("No hay reglas que cumplan los criterios de filtrado.")
                        else:
                            col_input, col_vacio = st.columns([1, 3])

                            with col_input:
                                num_filas = st.number_input(
                                    "Número de filas a mostrar",
                                    min_value=1,
                                    max_value=len(reglas_visibles),
                                    value=min(25, len(reglas_visibles)),
                                    step=1,
                                    key="num_filas_reglas"
                                )


                            tabla_reglas = reglas_visibles[columnas_a_mostrar].head(num_filas).rename(
                                columns={
                                    "support": "soporte",
                                    "confidence": "confianza"
                                }
                            )

                            st.dataframe(tabla_reglas, use_container_width=True)
                        

    # ---------------------------------
    # EVALUACIÓN DE REGLAS
    # ---------------------------------
    with st.expander("3. Evaluación de reglas", expanded=False):
        if "reglas_df" in st.session_state and not st.session_state["reglas_df"].empty:
            st.caption("Cálculo de las medidas seleccionadas sobre las reglas generadas.")

             # -------------------------
            # CONFIGURACIÓN EVALUACIÓN
            # -------------------------
            st.markdown("#### Configuración")

            col_input, col_vacio = st.columns([1, 3])

            with col_input:

                if "bloquear_evaluacion" not in st.session_state:
                    st.session_state["bloquear_evaluacion"] = False

                etiquetas_medidas_seleccionadas = st.multiselect(
                    "Selecciona las medidas a evaluar",
                    options=list(opciones_medidas.keys()),
                    default=[
                        "Soporte",
                        "Confianza",
                        "Lift",
                    ],
                    key="medidas_evaluacion",
                    disabled=st.session_state["bloquear_evaluacion"]
                )
                medidas_seleccionadas = [
                    opciones_medidas[etiqueta]
                    for etiqueta in etiquetas_medidas_seleccionadas
                ]

            if (
                st.session_state["bloquear_evaluacion"]
                and "evaluaciones" in st.session_state
            ):
                st.info("Medidas de evaluación bloqueadas.")

                if st.button("Modificar medidas de evaluación"):
                    st.session_state["bloquear_evaluacion"] = False
                    st.session_state["bloquear_filtrado"] = False

                    for clave in [
                        "evaluaciones",
                        "reglas_filtradas",
                    ]:
                        st.session_state.pop(clave, None)

                    st.rerun()

            st.markdown("---")
            st.markdown("#### Resultados")

            if st.button("Construir evaluaciones", use_container_width=True):
                try:
                    with st.spinner("Calculando medidas de interés. Este proceso puede tardar unos minutos..."):
                        evaluaciones = construir_evaluaciones_regla(
                            st.session_state["reglas_df"]
                        )

                except MemoryError:
                    st.error(
                        "No hay memoria suficiente para calcular las medidas de interés."
                    )
                    st.stop()

                except Exception as error:
                    st.error(
                        f"Se ha producido un error durante la evaluación de reglas: {error}"
                    )
                    st.stop()

                st.session_state["evaluaciones"] = evaluaciones
                st.session_state["bloquear_evaluacion"] = True
                st.session_state.pop("reglas_filtradas", None)

                st.rerun()

            if "evaluaciones" in st.session_state:
                st.write(f"Número de reglas evaluadas: {len(st.session_state['evaluaciones'])}")

                tabla_evaluaciones = construir_tabla_evaluaciones(
                    st.session_state["evaluaciones"],
                    medidas_seleccionadas
                )

                with st.expander("Ver evaluación de reglas", expanded=False):
                    if len(tabla_evaluaciones) == 0:
                        st.warning("No hay reglas que cumplan los criterios de filtrado.")
                    else:
                        col_input, col_vacio = st.columns([1, 3])

                        with col_input:
                            num_filas = st.number_input(
                                "Número de filas a mostrar",
                                min_value=1,
                                max_value=len(tabla_evaluaciones),
                                value=min(25, len(tabla_evaluaciones)),
                                step=1,
                                key="num_filas_evaluaciones"
                            )
                        st.dataframe(tabla_evaluaciones.head(num_filas), use_container_width=True)


    # ---------------------------------
    # FILTRADO DE REGLAS
    # ---------------------------------
    with st.expander("4. Filtrado de reglas", expanded=False):
        if "evaluaciones" in st.session_state and len(st.session_state["evaluaciones"]) > 0:
            st.caption("Selección de reglas según la medida y el umbral indicados.")

            # -------------------------
            # CONFIGURACIÓN FILTRADO
            # -------------------------
            st.markdown("#### Configuración")

            col_input, col_vacio = st.columns([1, 3])

            with col_input:
                if "bloquear_filtrado" not in st.session_state:
                    st.session_state["bloquear_filtrado"] = False
                etiquetas_medidas_filtrado = st.multiselect(
                    "Selecciona las medidas para filtrar",
                    options=list(opciones_medidas.keys()),
                    default=[],
                    disabled=st.session_state["bloquear_filtrado"]
                )

            criterios_filtrado = {}

            if etiquetas_medidas_filtrado:
                columnas_umbrales = st.columns(len(etiquetas_medidas_filtrado))

                for columna, etiqueta in zip(columnas_umbrales, etiquetas_medidas_filtrado):
                    medida = opciones_medidas[etiqueta]

                    with columna:
                        umbral = st.number_input(
                            f"Umbral para {etiqueta}",
                            value=0.0,
                            step=0.001,
                            format="%.3f",
                            key=f"umbral_filtrado_{medida}",
                            disabled=st.session_state["bloquear_filtrado"]
                        )

                    criterios_filtrado[medida] = umbral

            if (
                st.session_state["bloquear_filtrado"]
                and "reglas_filtradas" in st.session_state
            ):
                st.info("Criterios de filtrado bloqueados.")

                if st.button("Modificar criterios de filtrado"):
                    st.session_state["bloquear_filtrado"] = False
                    st.session_state.pop("reglas_filtradas", None)
                    st.rerun()

            st.markdown("---")
            st.markdown("#### Resultados")

            if st.button("Filtrar reglas", use_container_width=True):
                if not criterios_filtrado:
                    st.warning("Selecciona al menos una medida de filtrado.")
                else:
                    try:
                        with st.spinner("Filtrando reglas..."):
                            reglas_filtradas = filtrar_evaluaciones_regla_por_criterios(
                                st.session_state["evaluaciones"],
                                criterios=criterios_filtrado
                            )

                    except MemoryError:
                        st.error(
                            "No hay memoria suficiente para aplicar el filtrado."
                            " Prueba con un umbral más restrictivo."
                        )
                        st.stop()

                    except Exception as error:
                        st.error(
                            f"Se ha producido un error durante el filtrado de reglas: {error}"
                        )
                        st.stop()

                    st.session_state["reglas_filtradas"] = reglas_filtradas
                    st.session_state["medidas_filtrado"] = list(criterios_filtrado.keys())
                    st.session_state["criterios_filtrado"] = criterios_filtrado
                    st.session_state["bloquear_filtrado"] = True

                    st.rerun()

            if "reglas_filtradas" in st.session_state:
                st.write(f"Número de reglas filtradas: {len(st.session_state['reglas_filtradas'])}")

                medidas_filtrado = st.session_state.get("medidas_filtrado", list(criterios_filtrado.keys()))
                tabla_filtradas = construir_tabla_evaluaciones(
                    st.session_state["reglas_filtradas"],
                    medidas_filtrado
                )

                with st.expander("Ver reglas filtradas", expanded=False):
                    if len(tabla_filtradas) == 0:
                        st.warning("No hay reglas que cumplan los criterios de filtrado.")
                    else:
                        col_input, col_vacio = st.columns([1, 3])

                        with col_input:
                            num_filas = st.number_input(
                                "Número de filas a mostrar",
                                min_value=1,
                                max_value=len(tabla_filtradas),
                                value=min(25, len(tabla_filtradas)),
                                step=1,
                                key="num_filas_filtradas"
                            )

                        st.dataframe(
                            tabla_filtradas.head(num_filas),
                            use_container_width=True
                        )

    # ---------------------------------
    # ANÁLISIS DE REGLAS
    # ---------------------------------

    with st.expander("5. Resumen analítico", expanded=False):

        if "reglas_filtradas" in st.session_state:
            evaluaciones_analisis = st.session_state["reglas_filtradas"]
            origen_analisis = "reglas filtradas"
        elif "evaluaciones" in st.session_state:
            evaluaciones_analisis = st.session_state["evaluaciones"]
            origen_analisis = "reglas evaluadas"
        elif "reglas_df" in st.session_state:
            evaluaciones_analisis = construir_evaluaciones_regla(
                st.session_state["reglas_df"]
            )
            origen_analisis = "reglas generadas"
        else:
            evaluaciones_analisis = []

        if not evaluaciones_analisis:
            st.info("Construye primero las evaluaciones de las reglas para mostrar el resumen analítico.")
        else:
            tabla_analisis = construir_tabla_analisis(evaluaciones_analisis)

            st.caption(f"Resumen calculado sobre el conjunto de {origen_analisis}.")

            # Aseguramos formato numérico
            for medida in ["soporte", "confianza", "lift"]:
                if medida in tabla_analisis.columns:
                    tabla_analisis[medida] = pd.to_numeric(tabla_analisis[medida], errors="coerce")

            # Variables auxiliares para interpretar el lift
            if "lift" in tabla_analisis.columns:
                tabla_analisis["desviacion_lift"] = tabla_analisis["lift"] - 1

                tabla_analisis["tipo_dependencia"] = tabla_analisis["lift"].apply(
                    lambda x: "Independencia estadística"
                    if abs(x - 1) <= 0.01
                    else ("Dependencia positiva" if x > 1 else "Dependencia negativa")
                )

            # =====================================================
            # BLOQUE 1 + BLOQUE 2
            # =====================================================

            st.markdown("#### Resumen descriptivo")

            num_reglas = len(tabla_analisis)
            soporte_usado = st.session_state.get("soporte_minimo", None)
            medida_generacion_usada = st.session_state.get("medida_generacion", "-")
            umbral_generacion_usado = st.session_state.get("umbral_generacion", None)

            consecuente_mas_frecuente = (
                tabla_analisis["consecuente"].value_counts().idxmax()
                if "consecuente" in tabla_analisis.columns and len(tabla_analisis) > 0
                else "-"
            )

            criterios_filtrado_guardados = st.session_state.get("criterios_filtrado", {})

            if criterios_filtrado_guardados:
                filtros_texto = ", ".join(
                    [f"{medida} ≥ {umbral:.3f}" for medida, umbral in criterios_filtrado_guardados.items()]
                )
            else:
                filtros_texto = "Ninguno"

            resumen = {
                "Reglas analizadas": num_reglas,
                "Soporte mínimo": f"{soporte_usado:.3f}" if soporte_usado is not None else "-",
                "Medida de generación": medida_generacion_usada,
                "Umbral de generación": f"{umbral_generacion_usado:.3f}" if umbral_generacion_usado is not None else "-",
                "Soporte medio": f"{tabla_analisis['soporte'].mean():.3f}" if "soporte" in tabla_analisis.columns else "-",
                "Confianza media": f"{tabla_analisis['confianza'].mean():.3f}" if "confianza" in tabla_analisis.columns else "-",
                "Lift medio": f"{tabla_analisis['lift'].mean():.3f}" if "lift" in tabla_analisis.columns else "-",
                "Consecuente más frecuente": consecuente_mas_frecuente,
                "Filtros aplicados": filtros_texto,
            }

            st.dataframe(
                pd.DataFrame(resumen.items(), columns=["Indicador", "Valor"]),
                use_container_width=True,
                hide_index=True
            )
            

           
            # =====================================================
            # BLOQUE 2 + BLOQUE 3
            # =====================================================
            col_soporte, col_lift = st.columns(2)

            with col_soporte:
                st.markdown("#### Distribución del soporte")

                if "soporte" in tabla_analisis.columns:

                    grafico_soporte = (
                        alt.Chart(tabla_analisis)
                        .mark_bar()
                        .encode(
                            x=alt.X("soporte:Q", bin=True, title="Soporte"),
                            y=alt.Y("count():Q", title="Número de reglas"),
                            tooltip=[
                                alt.Tooltip("count():Q", title="Número de reglas")
                            ]
                        )
                        .properties(height=300)
                    )

                    st.altair_chart(grafico_soporte, use_container_width=True)

                else:
                    st.warning("No se encuentra la medida soporte.")

            with col_lift:
                st.markdown("#### Distribución de la medida lift")

                if "lift" in tabla_analisis.columns:
                    grafico_lift = (
                        alt.Chart(tabla_analisis)
                        .mark_bar()
                        .encode(
                            x=alt.X("lift:Q", bin=True, title="Lift"),
                            y=alt.Y("count():Q", title="Número de reglas"),
                            tooltip=[
                                alt.Tooltip("count():Q", title="Número de reglas")
                            ]
                        )
                        .properties(height=300)
                    )

                    linea_lift = (
                        alt.Chart(pd.DataFrame({"lift": [1]}))
                        .mark_rule(strokeDash=[6, 4])
                        .encode(x="lift:Q")
                    )

                    st.altair_chart(grafico_lift + linea_lift, use_container_width=True)

                    st.caption(
                        "La línea vertical marca lift = 1, que corresponde al caso de independencia. "
                        "Valores superiores a 1 indican dependencia positiva, mientras que valores "
                        "inferiores a 1 indican dependencia negativa."
                    )
                else:
                    st.warning("No se encuentra la medida lift.")

            # =====================================================
            # BLOQUE 4: gráfico + tabla
            # =====================================================
            col_scatter, col_tabla = st.columns([1.2, 1])
            with col_scatter:
                st.markdown("#### Confianza frente a lift")

                if {"soporte", "confianza", "lift"}.issubset(tabla_analisis.columns):

                    seleccion = alt.selection_point(
                        name="seleccion_regla",
                        fields=["id_regla"],
                        empty="all",
                        toggle=True
                    )

                    puntos = (
                        alt.Chart(tabla_analisis)
                        .mark_circle(opacity=0.75)
                        .encode(
                            x=alt.X("lift:Q", title="Lift"),
                            y=alt.Y("confianza:Q", title="Confianza"),
                            size=alt.Size(
                                "soporte:Q",
                                title="Soporte",
                                legend=alt.Legend(
                                    titleFontSize=12,
                                    labelFontSize=10
                                )
                            ),
                            color=alt.Color(
                                "desviacion_lift:Q",
                                title="Lift",
                                scale=alt.Scale(
                                    scheme="redblue",
                                    domainMid=0
                                ),
                                legend=alt.Legend(
                                    orient="right",
                                    titleFontSize=12,
                                    labelFontSize=10,
                                    gradientLength=90
                                )
                            ),
                            tooltip=[
                                alt.Tooltip("regla:N", title="Regla"),
                                alt.Tooltip("soporte:Q", title="Soporte", format=".3f"),
                                alt.Tooltip("confianza:Q", title="Confianza", format=".3f"),
                                alt.Tooltip("lift:Q", title="Lift", format=".3f"),
                                alt.Tooltip("tipo_dependencia:N", title="Tipo"),
                            ],
                            opacity=alt.condition(seleccion, alt.value(0.9), alt.value(0.18))
                        )
                        .add_params(seleccion)
                    )

                    linea_independencia = (
                        alt.Chart(pd.DataFrame({"lift": [1]}))
                        .mark_rule(strokeDash=[6, 4])
                        .encode(x="lift:Q")
                    )

                    evento = st.altair_chart(
                        (puntos + linea_independencia).properties(height=350),
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="seleccion_regla"
                    )
                    st.caption(
                            "Cada punto representa una regla. El eje horizontal muestra la medida lift, "
                            "el eje vertical la confianza y el tamaño del punto el soporte. "
                            "Las reglas más alejadas de lift = 1 presentan una mayor desviación "
                            "respecto a la independencia."
                        )

                    ids_seleccionados = []

                    if evento.selection and "seleccion_regla" in evento.selection:
                        puntos_seleccionados = evento.selection["seleccion_regla"]

                        ids_seleccionados = [
                            punto["id_regla"]
                            for punto in puntos_seleccionados
                            if "id_regla" in punto
                        ]

                    if ids_seleccionados:
                        tabla_scatter = tabla_analisis[
                            tabla_analisis["id_regla"].isin(ids_seleccionados)
                        ]
                    else:
                        tabla_scatter = tabla_analisis

                    with col_tabla:
                        st.markdown("#### Reglas seleccionadas")

                        columnas_tabla = [
                            "regla",
                            "soporte",
                            "confianza",
                            "lift",
                            "tipo_dependencia"
                        ]

                        st.dataframe(
                            tabla_scatter[columnas_tabla],
                            use_container_width=True,
                            hide_index=True
                        )

                    

                else:
                    st.warning(
                        "Para mostrar este gráfico es necesario que las reglas tengan calculadas "
                        "las medidas soporte, confianza y lift."
                    )
                

            # =====================================================
            # BLOQUE 5: botón de generar informe
            # =====================================================
            
            col1,col2,col3,col4= st.columns([2,1,1,2])

            with col2:
                num_reglas_informe = st.number_input(
                    "Número máximo de reglas a incluir en el informe",
                    min_value=1,
                    max_value=len(tabla_analisis),
                    value=min(500, len(tabla_analisis)),
                    step=1,
                    key="num_reglas_informe"
                )
            informe_pdf = generar_informe_pdf(
                tabla_analisis=tabla_analisis,
                resumen=resumen,
                origen_analisis=origen_analisis,
                max_reglas=num_reglas_informe
            )
            with col3:
                st.caption(" ")
                st.caption(" ")
                st.caption(" ")
                st.download_button(
                    label="Descargar informe PDF",
                    data=informe_pdf,
                    file_name="informe_reglas_asociacion.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
