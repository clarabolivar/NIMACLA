from abc import ABC, abstractmethod
import pandas as pd


# =============================================================
# CLASES
# =============================================================

# --------------------------------------------------
# Clase: InterpretacionTabular
# --------------------------------------------------
# Clase abstracta que representa una interpretación tabular.
# Modela formalmente una interpretación Φ = (ΦI, ΦT), donde:
#
#   - ΦI construye el conjunto de ítems a partir del esquema
#   - ΦT transforma cada tupla en una transacción
#
# Esta clase define la interfaz que deben implementar
# todas las interpretaciones.
#
# Métodos:
#   - construir_items: construcción del conjunto de ítems (ΦI)
#   - construir_transacciones: construcción de la base transaccional (ΦT)
#
class InterpretacionTabular(ABC):

    @abstractmethod
    def construir_items(self, datos: pd.DataFrame) -> set[str]:
        pass

    @abstractmethod
    def construir_transacciones(self, datos: pd.DataFrame) -> list[list[str]]:
        pass


# --------------------------------------------------
# Clase: InterpretacionAtributoValor
# --------------------------------------------------
# Implementa la interpretación atributo–valor.
# En esta interpretación:
#
#   - Cada ítem se define como un par (A, v), donde A es un atributo
#     y v es un valor observado en su dominio.
#
#   - Cada tupla genera exactamente un ítem por atributo (si no es nulo).
#
# Formalmente:
#   - ΦI = {(A, v) : v ∈ Dom(A) observado}
#   - ΦT(t) = {(A1, v1), ..., (An, vn)}
#
# Representación:
#   - Los ítems se codifican como cadenas del tipo "A=v"
#

class InterpretacionAtributoValor(InterpretacionTabular):

    def construir_items(self, datos: pd.DataFrame) -> set[str]:
        items = set()

        for atributo in datos.columns:
            for valor in datos[atributo].dropna().unique():
                items.add(self._formatear_item(atributo, valor))

        return items

    def construir_transacciones(self, datos: pd.DataFrame) -> list[list[str]]:
        transacciones = []

        for _, tupla in datos.iterrows():
            transaccion = []

            for atributo, valor in tupla.items():
                if pd.notna(valor):
                    transaccion.append(self._formatear_item(atributo, valor))

            transacciones.append(transaccion)

        return transacciones

    def _formatear_item(self, atributo: str, valor) -> str:
        return f"{atributo}={valor}"
    

# --------------------------------------------------
# Clase: InterpretacionAtributoGrupoValores
# --------------------------------------------------
# Implementa la interpretación atributo–grupo de valores.
# En esta interpretación:
#
#   - Cada ítem se define como un par (A, G), donde G es un subconjunto
#     del dominio de A (por ejemplo, un intervalo o grupo de valores).
#
#   - Una tupla puede generar múltiples ítems por atributo, en función
#     de los grupos a los que pertenezca su valor.
#
# Formalmente:
#   - phi_I = {(A, G) : G ⊆ Dom(A)}
#   - phi_T(t) = {(A, G) : v ∈ G}
#
# Configuración:
#   - Requiere un diccionario que define los grupos para cada atributo:
#
#     grupos_por_atributo = {
#         "A": [
#             {
#                 "nombre": "G1",
#                 "tipo": "categorico",
#                 "valores": {"v1", "v2", ...}
#             },
#             {
#                 "nombre": "G2",
#                 "tipo": "intervalo",
#                 "minimo": a,
#                 "maximo": b,
#                 "cerrado_izquierda": True,
#                 "cerrado_derecha": False
#             },
#             ...
#         ]
#     }
#
#   - Las definiciones pueden ser:
#       - grupos categóricos definidos por conjuntos de valores.
#       - intervalos numéricos, con extremos abiertos o cerrados.
#
class InterpretacionAtributoGrupoValores(InterpretacionTabular):

    def __init__(self, grupos_por_atributo: dict):
        self.grupos_por_atributo = grupos_por_atributo

    # --------------------------------------------------
    # Método: construir_items
    # --------------------------------------------------
    # Construye el conjunto de ítems asociado a la interpretación.
    #
    # A partir de la configuración de grupos definida para cada atributo,
    # genera todos los ítems posibles de la forma (A, G), donde G es un grupo
    # de valores del dominio de A.
    #
    # @param datos pd.DataFrame:
    #   DataFrame tabular que contiene las tuplas originales.
    #
    # @return set[str]:
    #   Conjunto de ítems representados como cadenas en formato "A=G".
    #
    def construir_items(self, datos: pd.DataFrame) -> set[str]:
        items = set()

        for atributo, grupos in self.grupos_por_atributo.items():
            for definicion_grupo in grupos:
                items.add(self._formatear_item(atributo, definicion_grupo))

        return items

    # --------------------------------------------------
    # Método: construir_transacciones
    # --------------------------------------------------
    # Construye la base de datos transaccional asociada a la interpretación.
    #
    # Para cada tupla del dataset, determina los grupos a los que pertenece
    # el valor de cada atributo y genera los ítems correspondientes.
    # Una misma tupla puede generar múltiples ítems por atributo.
    #
    # @param datos pd.DataFrame:
    #   DataFrame tabular que contiene las tuplas originales.
    #
    # @return list[list[str]]:
    #   Lista de transacciones, donde cada transacción es una lista de ítems
    #   representados como cadenas en formato "A=G".
    #
    def construir_transacciones(self, datos: pd.DataFrame) -> list[list[str]]:
        transacciones = []

        for _, tupla in datos.iterrows():
            transaccion = []

            for atributo, valor in tupla.items():
                if pd.isna(valor):
                    continue

                if atributo not in self.grupos_por_atributo:
                    continue

                grupos = self.grupos_por_atributo[atributo]

                for definicion_grupo in grupos:
                    if self._valor_pertenece_a_grupo(valor, definicion_grupo):
                        transaccion.append(
                            self._formatear_item(atributo, definicion_grupo)
                        )

            transacciones.append(transaccion)

        return transacciones

    # --------------------------------------------------
    # Método: _valor_pertenece_a_grupo
    # --------------------------------------------------
    # Determina si un valor pertenece a un grupo dado.
    #
    # Evalúa la pertenencia de un valor a una definición de grupo,
    # distinguiendo entre grupos categóricos e intervalos numéricos.
    #
    # @param valor:
    #   Valor que se desea comprobar.
    #
    # @param definicion_grupo dict:
    #   Diccionario que define el grupo. Debe contener una clave "tipo",
    #   cuyo valor puede ser:
    #     - "categorico": grupo definido por un conjunto de valores.
    #     - "intervalo": grupo definido por un intervalo numérico.
    #
    # @return bool:
    #   True si el valor pertenece al grupo, False en caso contrario.
    #
    # Nota:
    #   - En los intervalos se permite distinguir entre extremos abiertos
    #     y cerrados mediante las claves "cerrado_izquierda" y
    #     "cerrado_derecha".
    #
    def _valor_pertenece_a_grupo(self, valor, definicion_grupo) -> bool:
        if definicion_grupo["tipo"] == "categorico":
            return str(valor) in definicion_grupo["valores"]

        if definicion_grupo["tipo"] == "intervalo":
            valor = float(valor)

            minimo = definicion_grupo["minimo"]
            maximo = definicion_grupo["maximo"]

            cumple_izquierda = (
                valor >= minimo
                if definicion_grupo["cerrado_izquierda"]
                else valor > minimo
            )

            cumple_derecha = (
                valor <= maximo
                if definicion_grupo["cerrado_derecha"]
                else valor < maximo
            )

            return cumple_izquierda and cumple_derecha

        return False

    # --------------------------------------------------
    # Método: _formatear_definicion_grupo
    # --------------------------------------------------
    # Genera la representación completa de la definición de un grupo.
    # Para grupos categóricos muestra el conjunto de valores que lo 
    # forman; para grupos numéricos muestra el intervalo correspondiente.
    #
    # @param atributo str:
    #   Nombre del atributo.
    #
    # @param definicion_grupo dict:
    #   Diccionario que define el grupo de valores.
    #
    # @return str:
    #   Cadena de texto en formato "A={v1, v2}" para grupos categóricos
    #   o "A=[a,b]", "A=(a,b]", etc. para intervalos numéricos.
    #
    def _formatear_definicion_grupo(self, atributo: str, definicion_grupo) -> str: 
        if definicion_grupo["tipo"] == "categorico":
            valores = sorted(definicion_grupo["valores"])
            return f"{atributo}={{" + ", ".join(valores) + "}}"

        if definicion_grupo["tipo"] == "intervalo":
            izq = "[" if definicion_grupo["cerrado_izquierda"] else "("
            der = "]" if definicion_grupo["cerrado_derecha"] else ")"

            minimo = definicion_grupo["minimo"]
            maximo = definicion_grupo["maximo"]

            return f"{atributo}={izq}{minimo},{maximo}{der}"
    
    # --------------------------------------------------
    # Método: _formatear_item
    # --------------------------------------------------
    # Genera la representación de un ítem asociado a un grupo.
    #
    # @param atributo str:
    #   Nombre del atributo.
    #
    # @param definicion_grupo dict:
    #   Diccionario que define el grupo de valores.
    #
    # @return str:
    #   Cadena de texto en formato "Atributo=NombreGrupo".
    def _formatear_item(self, atributo: str, definicion_grupo) -> str: 
        return f"{atributo}={definicion_grupo['nombre']}"
    
