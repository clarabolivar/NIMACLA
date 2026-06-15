# =============================================================
# =============================================================
# FUNCIÓN QUE IMPLEMENTA EL ALGORITMO FP-GROWTH
# =============================================================
# =============================================================

from mlxtend.frequent_patterns import fpgrowth
import multiprocessing as mp

# --------------------------------------------------
# Función: _ejecutar_fpgrowth
# --------------------------------------------------
# Función auxiliar que ejecuta el algoritmo FP-Growth
# en un proceso independiente.
#
# El resultado obtenido se envía mediante una cola
# entre procesos. Si la ejecución finaliza
#  correctamente, se almacena el DataFrame
# de itemsets frecuentes. Si se produce una excepción,
# se almacena el mensaje de error.
#
# @param matriz_codificada pd.DataFrame:
#   DataFrame booleano con una columna por ítem y una fila
#   por transacción.
#
# @param soporte_minimo float:
#   Umbral mínimo de soporte utilizado por FP-Growth.
#
# @param cola multiprocessing.Queue:
#   Cola utilizada para devolver el resultado o el error
#   al proceso principal.
#
# @return None
#
def _ejecutar_fpgrowth(matriz_codificada, soporte_minimo, cola):
    try:
        itemsets = fpgrowth(
            matriz_codificada,
            min_support=soporte_minimo,
            use_colnames=True
        )
        cola.put(("ok", itemsets))

    except Exception as error:
        cola.put(("error", str(error)))


# --------------------------------------------------
# Función: algoritmo_fpgrowth
# --------------------------------------------------
# Ejecuta el algoritmo FP-Growth sobre una matriz booleana
# de transacciones para obtener los itemsets frecuentes.
#
# @param matriz_codificada pd.DataFrame:
#   DataFrame booleano con una columna por ítem y una fila
#   por transacción.
#
# @param soporte_minimo float:
#   Umbral mínimo de soporte.
#
# @param tiempo_maximo int:
#   Tiempo máximo de ejecución permitido, expresado en segundos.
#   Si se supera este límite, se interrumpe el proceso y se lanza
#   una excepción TimeoutError.
#
# @return pd.DataFrame:
#   DataFrame que contiene los itemsets frecuentes y su soporte.
#
def algoritmo_fpgrowth(
    matriz_codificada,
    soporte_minimo: float,
    tiempo_maximo: int = 300
):
    cola = mp.Queue()

    proceso = mp.Process(
        target=_ejecutar_fpgrowth,
        args=(matriz_codificada, soporte_minimo, cola)
    )

    proceso.start()
    proceso.join(timeout=tiempo_maximo)

    if proceso.is_alive():
        proceso.terminate()
        proceso.join()
        raise TimeoutError()

    estado, resultado = cola.get()

    if estado == "error":
        raise RuntimeError(resultado)

    return resultado


