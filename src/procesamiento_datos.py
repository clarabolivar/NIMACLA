# =============================================================
# =============================================================
# FUNCIONES PARA LA TRANSFORMACIÓN DE DATOS
#
# A partir de una lista de transacciones generada mediante una
# interpretación tabular, se construye una matriz booleana con
# formato tabular. Esta representación es la que se utiliza
# para aplicar los algoritmos de generación de itemsets
# frecuentes.
# =============================================================
# =============================================================

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


# --------------------------------------------------
# Función: codificar_transacciones
# --------------------------------------------------
# Convierte una lista de transacciones en una matriz booleana.
#
# @param transacciones list[list[str]]:
#   Lista de transacciones, donde cada transacción se representa
#   como una lista de ítems.
#
# @return pd.DataFrame:
#   DataFrame booleano con una fila por transacción y una
#   columna por ítem, donde cada valor indica la presencia o
#   ausencia del ítem en la transacción.
def codificar_transacciones(transacciones: list[list[str]]) -> pd.DataFrame:
    
    if not transacciones:
        raise ValueError("La lista de transacciones no puede estar vacía.")

    codificador = TransactionEncoder()
    matriz_codificada = codificador.fit(transacciones).transform(transacciones)
    datos_codificados = pd.DataFrame(
        matriz_codificada,
        columns=codificador.columns_
    )
    return datos_codificados