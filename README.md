# NIMACLA

NIMACLA es una herramienta para el análisis de reglas de asociación sobre conjuntos de datos tabulares. La aplicación permite cargar un fichero CSV, transformar los datos mediante distintas interpretaciones tabulares, extraer itemsets frecuentes, generar reglas de asociación, evaluarlas mediante diferentes medidas de interés y filtrar los resultados obtenidos. Además, realiza un resumen analítico y permite generar un informe en formato pdf sobre las reglas obtenidas.

La herramienta ha sido desarrollada como parte del Trabajo de Fin de Grado del Doble Grado en Ingeniería Informática y Matemáticas.

## Funcionalidades

- Carga de conjuntos de datos tabulares en formato CSV.
- Transformación del conjunto de datos mediante distintas interpretaciones tabulares.
- Extracción de itemsets frecuentes mediante el algoritmo FP-Growth.
- Generación de reglas de asociación.
- Evaluación de reglas mediante distintas medidas de interés:
  - soporte,
  - confianza,
  - lift,
  - leverage,
  - conviction,
  - factor de certeza,
  - información mutua.
- Filtrado de reglas mediante varios criterios simultáneos.
- Generación de un resumen analítico.
- Generación de un informe en formato PDF.

## Instalación

Se recomienda utilizar un entorno virtual de Python.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
