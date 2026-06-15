# =============================================================
# =============================================================
# FUNCIONES QUE CALCULAN DISTINTAS MEDIDAS DE INTERÉS 
# =============================================================
# =============================================================

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.reglas import EstadisticasRegla


# =============================================================
# MEDIDAS
# =============================================================

# --------------------------------------------------
# Función: medida_soporte
# --------------------------------------------------
# Calcula el soporte de una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor del soporte.
#
def medida_soporte(estadisticas: "EstadisticasRegla") -> float:
    return estadisticas.p_xy


# --------------------------------------------------
# Función: medida_confianza
# --------------------------------------------------
# Calcula la confianza de una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor de confianza.
#
def medida_confianza(estadisticas: "EstadisticasRegla") -> float:
    p_x = estadisticas.p_xy + estadisticas.p_x_no_y

    if p_x == 0:
        return 0.0

    return estadisticas.p_xy / p_x


# --------------------------------------------------
# Función: medida_lift
# --------------------------------------------------
# Calcula el lift de una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor del lift.
#
def medida_lift(estadisticas: "EstadisticasRegla") -> float:
    p_x = estadisticas.p_xy + estadisticas.p_x_no_y
    p_y = estadisticas.p_xy + estadisticas.p_no_x_y

    if p_x == 0 or p_y == 0:
        return float("nan")

    return estadisticas.p_xy / (p_x * p_y)


# --------------------------------------------------
# Función: medida_leverage
# --------------------------------------------------
# Calcula el valor de leverage de una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor del leverage.
#
def medida_leverage(estadisticas: "EstadisticasRegla") -> float:
    p_x = estadisticas.p_xy + estadisticas.p_x_no_y
    p_y = estadisticas.p_xy + estadisticas.p_no_x_y

    return estadisticas.p_xy - p_x * p_y


# --------------------------------------------------
# Función: medida_conviction
# --------------------------------------------------
# Calcula el valor de conviction de una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor de la conviction. 
#
def medida_conviction(estadisticas: "EstadisticasRegla") -> float:
    p_y = estadisticas.p_xy + estadisticas.p_no_x_y
    confianza = medida_confianza(estadisticas)

    if confianza == 1.0:
        return float("inf")

    denominador = 1.0 - confianza
    numerador = 1.0 - p_y

    if denominador == 0:
        return float("inf")

    return numerador / denominador


# --------------------------------------------------
# Función: medida_factor_certeza
# --------------------------------------------------
# Calcula el factor de certeza de una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor del factor de certeza.
#
def medida_factor_certeza(estadisticas: "EstadisticasRegla") -> float:
    p_y = estadisticas.p_xy + estadisticas.p_no_x_y
    confianza = medida_confianza(estadisticas)

    if confianza >= p_y:
        denominador = 1.0 - p_y
    else:
        denominador = p_y

    if denominador == 0:
        return 0.0

    return (confianza - p_y) / denominador


# --------------------------------------------------
# Función: termino_informacion_mutua
# --------------------------------------------------
# Calcula un término de la suma de la medida información mutua.
#
# @param p_xy float:
#   Probabilidad conjunta.
#
# @param p_x float:
#   Probabilidad marginal de X.
#
# @param p_y float:
#   Probabilidad marginal de Y.
#
# @return float:
#   Valor correspondiente de la suma. 
#
def termino_informacion_mutua(p_xy: float, p_x: float, p_y: float) -> float:
    if p_xy <= 0:
        return 0.0

    denominador = p_x * p_y

    if denominador <= 0:
        return 0.0

    razon = p_xy / denominador

    if razon <= 0:
        return 0.0

    return p_xy * math.log2(razon)


# --------------------------------------------------
# Función: medida_informacion_mutua
# --------------------------------------------------
# Calcula la información mutua asociada a una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return float:
#   Valor de la información mutua.
#
def medida_informacion_mutua(estadisticas: "EstadisticasRegla") -> float:
    a = estadisticas.p_xy
    b = estadisticas.p_x_no_y
    c = estadisticas.p_no_x_y
    d = estadisticas.p_no_x_no_y

    p_x = a + b
    p_no_x = c + d
    p_y = a + c
    p_no_y = b + d

    return (
        termino_informacion_mutua(a, p_x, p_y)
        + termino_informacion_mutua(b, p_x, p_no_y)
        + termino_informacion_mutua(c, p_no_x, p_y)
        + termino_informacion_mutua(d, p_no_x, p_no_y)
    )


# --------------------------------------------------
# Función: calcular_medidas
# --------------------------------------------------
# Calcula el conjunto completo de medidas implementadas
# para una regla.
#
# @param estadisticas EstadisticasRegla:
#   Tabla de contingencia asociada a la regla.
#
# @return dict[str, float]:
#   Diccionario con los valores de las medidas calculadas.
#
def calcular_medidas(estadisticas: "EstadisticasRegla") -> dict[str, float]:
    return {
        "soporte": medida_soporte(estadisticas),
        "confianza": medida_confianza(estadisticas),
        "lift": medida_lift(estadisticas),
        "leverage": medida_leverage(estadisticas),
        "conviction": medida_conviction(estadisticas),
        "factor_certeza": medida_factor_certeza(estadisticas),
        "informacion_mutua": medida_informacion_mutua(estadisticas),
    }