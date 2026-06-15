# =============================================================
# =============================================================
# FUNCIONES Y CLASES RELACIONADAS CON LAS REGLAS DE ASOCIACIÓN
# =============================================================
# =============================================================

from dataclasses import dataclass
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from src.medidas import calcular_medidas


# =============================================================
# CLASES
# =============================================================

# --------------------------------------------------
# Clase: Regla
# --------------------------------------------------
# Representa una regla de asociación de la forma X => Y.
#
# Atributos:
#   - antecedente (frozenset[str])
#   - consecuente (frozenset[str])
#
# Nota:
#   - Se utiliza frozenset para garantizar inmutabilidad.
@dataclass(frozen=True)
class Regla:
    antecedente: frozenset[str]
    consecuente: frozenset[str]

    # --------------------------------------------------
    # Método: __str__
    # --------------------------------------------------
    # Devuelve una representación de la regla.
    #
    # @return str: 
    #   Cadena de texto en formato "{antecedente} => {consecuente}".
    #
    def __str__(self) -> str:
        antecedente_str = ", ".join(sorted(self.antecedente))
        consecuente_str = ", ".join(sorted(self.consecuente))
        return f"{{{antecedente_str}}} => {{{consecuente_str}}}"

    # --------------------------------------------------
    # Método: __repr__
    # --------------------------------------------------
    # Devuelve una representación de la regla.
    #
    # @return str: 
    #   Cadena de texto en formato "{antecedente} => {consecuente}".
    #
    # Nota:
    #   - Es la misma salida que la de __str__. Se utiliza en contextos internos.
    def __repr__(self) -> str:
        return self.__str__()


# --------------------------------------------------
# Clase: EstadisticasRegla
# --------------------------------------------------
# Representa la tabla de contingencia asociada a una regla.
#
# Atributos:
#   - p_xy (float): probabilidad de que aparezcan conjuntamente antecedente y consecuente.
#   - p_x_no_y (float): probabilidad de que aparezca el antecedente sin el consecuente.
#   - p_no_x_y (float):  probabilidad de que aparezca el consecuente sin el antecedente.
#   - p_no_x_no_y (float): probabilidad de que no aparezca ni el antecedente ni el consecuente.
#
@dataclass(frozen=True)
class EstadisticasRegla:
    p_xy: float
    p_x_no_y: float
    p_no_x_y: float
    p_no_x_no_y: float


# --------------------------------------------------
# Clase: EvaluacionRegla
# --------------------------------------------------
# Representa la evaluación completa de una regla. Integra:
#   - la regla de asociación,
#   - su tabla de contingencia,
#   - y las medidas calculadas a partir de ella.
#
# Atributos:
#   - regla (Regla): regla de asociación evaluada.
#   - estadisticas (EstadisticasRegla): tabla de contingencia asociada a la regla.
#   - medidas (dict[str, float]): diccionario con los valores de las medidas de interés.
#
@dataclass
class EvaluacionRegla:
    regla: Regla
    estadisticas: EstadisticasRegla
    medidas: dict[str, float]

    # --------------------------------------------------
    # Método: __str__
    # --------------------------------------------------
    # Devuelve una representación de la evaluación.
    #
    # @return str: 
    #   Cadena estructurada con regla, estadísticas y medidas.
    #
    def __str__(self) -> str:
        return (
            f"EvaluacionRegla(\n"
            f"  regla={self.regla},\n"
            f"  estadisticas={self.estadisticas},\n"
            f"  medidas={self.medidas}\n"
            f")"
        )

    # --------------------------------------------------
    # Método: __repr__
    # --------------------------------------------------
    # Devuelve una representación de la evaluación.
    #
    # @return str: 
    #   Cadena estructurada con regla, estadísticas y medidas.
    #
    # Nota:
    #   - Es la misma salida que la de __str__. Se utiliza en contextos internos.
    def __repr__(self) -> str:
        return self.__str__()


# =============================================================
# FUNCIONES
# =============================================================

# --------------------------------------------------
# Función: construir_regla
# --------------------------------------------------
# Construye un objeto de tipo Regla a partir de una fila
# del DataFrame de reglas.
#
# @param fila (pd.Series):
#   Fila del DataFrame que contiene, al menos, las columnas
#   "antecedents" y "consequents".
#
# @return Regla:
#   Objeto Regla construido a partir de la información de la fila.
def construir_regla(fila: pd.Series) -> Regla:
    return Regla(
        antecedente=frozenset(fila["antecedents"]),
        consecuente=frozenset(fila["consequents"])
    )


# --------------------------------------------------
# Función: construir_estadisticas_regla
# --------------------------------------------------
# Construye un objeto de tipo EstadisticasRegla a partir
# de una fila del DataFrame de reglas.
# Utilizando las probabilidades básicas disponibles en la fila
# ("support", "antecedent support" y "consequent support"),
# reconstruye la tabla de contingencia completa asociada a la regla.
#
# @param fila pd.Series:
#   Fila del DataFrame que contiene, al menos, las columnas
#   "support", "antecedent support" y "consequent support".
#
# @return EstadisticasRegla:
#   Objeto que contiene las cuatro probabilidades de la tabla
#   de contingencia asociada a la regla.
#
def construir_estadisticas_regla(fila: pd.Series) -> EstadisticasRegla:
    p_xy = float(fila["support"])
    p_x = float(fila["antecedent support"])
    p_y = float(fila["consequent support"])

    p_x_no_y = p_x - p_xy
    p_no_x_y = p_y - p_xy
    p_no_x_no_y = 1.0 - p_x - p_y + p_xy

    return EstadisticasRegla(
        p_xy=limpiar_probabilidad(p_xy),
        p_x_no_y=limpiar_probabilidad(p_x_no_y),
        p_no_x_y=limpiar_probabilidad(p_no_x_y),
        p_no_x_no_y=limpiar_probabilidad(p_no_x_no_y),
    )


# --------------------------------------------------
# Función: construir_evaluacion_regla
# --------------------------------------------------
# Construye un objeto de tipo EvaluacionRegla a partir
# de una fila del DataFrame de reglas. Sus funciones son:
#   1. construye la regla,
#   2. reconstruye su tabla de contingencia,
#   3. calcula las medidas de interés a partir de dicha tabla.
#
# @param fila pd.Series:
#   Fila del DataFrame de reglas.
#
# @return EvaluacionRegla:
#   Objeto que integra regla, estadísticas y medidas.
def construir_evaluacion_regla(fila: pd.Series) -> EvaluacionRegla:
    regla = construir_regla(fila)
    estadisticas = construir_estadisticas_regla(fila)
    medidas = calcular_medidas(estadisticas)

    return EvaluacionRegla(
        regla=regla,
        estadisticas=estadisticas,
        medidas=medidas
    )


# --------------------------------------------------
# Función: construir_evaluaciones_regla
# --------------------------------------------------
# Construye la lista completa de evaluaciones de regla
# a partir del DataFrame de reglas.
#
# @param reglas_df pd.DataFrame:
#   DataFrame que contiene las reglas generadas.
#
# @return list[EvaluacionRegla]:
#   Lista de evaluaciones completas, una por cada regla.
#
def construir_evaluaciones_regla(reglas_df: pd.DataFrame) -> list[EvaluacionRegla]:
    evaluaciones = []

    for _, fila in reglas_df.iterrows():
        evaluacion = construir_evaluacion_regla(fila)
        evaluaciones.append(evaluacion)

    return evaluaciones


# --------------------------------------------------
# Función: limpiar_probabilidad
# --------------------------------------------------
# Corrige pequeños errores numéricos en probabilidades.
#
# @param valor float:
#   Valor de probabilidad que se quiere corregir.
#
# @param eps float:
#   Tolerancia utilizada para decidir si un valor se considera
#   suficientemente cercano a 0 o a 1.
#
# @return float:
#   Valor corregido. Si está cerca de 0, devuelve 0.0; si está cerca
#   de 1, devuelve 1.0; en caso contrario, devuelve el valor original.
#
def limpiar_probabilidad(valor: float, eps: float = 1e-12) -> float:
    if abs(valor) < eps:
        return 0.0

    if abs(valor - 1.0) < eps:
        return 1.0

    return valor



# --------------------------------------------------
# Función: filtrar_evaluaciones_regla_por_criterios
# --------------------------------------------------
# Filtra un conjunto de evaluaciones de regla según múltiples medidas
# y sus respectivos umbrales.
# Recorre la lista de evaluaciones y selecciona únicamente aquellas
# reglas que satisfacen simultáneamente todos los criterios indicados.
#
# @param evaluaciones list[EvaluacionRegla]:
#   Lista de evaluaciones de regla sobre la que se aplica el filtrado.
#
# @param criterios dict[str, float]:
#   Diccionario que asocia a cada medida el umbral mínimo que debe
#   satisfacer. Las claves representan los nombres de las medidas y
#   los valores los umbrales correspondientes.
#
# @return list[EvaluacionRegla]:
#   Lista de evaluaciones que satisfacen simultáneamente todos los
#   criterios de filtrado especificados.
#
def filtrar_evaluaciones_regla_por_criterios(
    evaluaciones: list[EvaluacionRegla],
    criterios: dict[str, float]
) -> list[EvaluacionRegla]:
    evaluaciones_filtradas = []

    for evaluacion in evaluaciones:
        cumple_criterios = True

        for nombre_medida, umbral in criterios.items():
            valor = evaluacion.medidas.get(nombre_medida)

            if valor is None or pd.isna(valor) or valor < umbral:
                cumple_criterios = False
                break

        if cumple_criterios:
            evaluaciones_filtradas.append(evaluacion)

    return evaluaciones_filtradas


# --------------------------------------------------
# Función: generar_reglas
# --------------------------------------------------
# Genera reglas de asociación que cumplen un umbral mínimo
# respecto a una medida dada a partir de itemsets frecuentes.
#
# @param itemsets_frecuentes pd.DataFrame:
#   DataFrame que contiene los itemsets frecuentes y su soporte.
#
# @param medida str:
#   Nombre de la medida utilizada como criterio de filtrado.
#
# @param umbral_minimo float:
#   Valor mínimo que debe alcanzar la medida para que la regla sea
#   generada.
#
# @return pd.DataFrame:
#   DataFrame con las reglas generadas y la información básica necesaria 
#   para su posterior evaluación.
#
def generar_reglas(
    itemsets_frecuentes: pd.DataFrame,
    medida: str = "confidence",
    umbral_minimo: float = 0.0
) -> pd.DataFrame:

    reglas_df = association_rules(
        itemsets_frecuentes,
        metric=medida,
        min_threshold=umbral_minimo
    )

    return reglas_df