# Modelado híbrido del Modelo Mínimo de Bergman para la dinámica glucosa-insulina

**Autora:** Laura Llorente Martínez

Este repositorio contiene el código desarrollado como parte de mi Trabajo de Fin de Grado (TFG) de Ingeniería del Software titulado **«Modelado híbrido del Modelo Mínimo de Bergman para la dinámica glucosa-insulina»**. 

Todo el código se ha implementado en el lenguaje **Python**, haciendo uso de librerías de aprendizaje automático y de procesamiento de datos.

El objetivo principal del presente TFG es diseñar, desarrollar, entrenar y evaluar un modelo híbrido basado en **ecuaciones diferenciales universales**, con la finalidad de estimar el parámetro que determina la capacidad de la insulina para aumentar la captación de la glucosa ($p_3$) y mejorar las predicciones de los niveles de glucosa en pacientes que padecen diabetes. Para conseguir el enfoque híbrido, se combina una red neuronal profunda con el Modelo fisiológico Mínimo de Bergman.

Asimismo, se realiza una comparación con la resolución numérica del modelo clásico de Bergman, para cuantificar la mejora predictiva, y evaluar la robustez y el cumplimiento de las restricciones del modelo híbrido.

---

## Conjunto de datos

La fuente principal de datos para el trabajo proviene del simulador **simglucose** [^3]. El paquete `simglucose` es una implementación en Python del simulador de diabetes tipo $1$ desarrollado por las universidades de Virginia y Padova. Cuenta con la validación de la Administración de Alimentos y Medicamentos de los Estados Unidos como sustituto aceptado para los ensayos preclínicos para probar tratamientos con insulina y evaluar algoritmos de control glucémico.

Para este trabajo, se ha elegido la simulación de todos los pacientes disponibles, es decir, $30$ sujetos distribuidos en $10$ adultos, $10$ adolescentes y $10$ niños. Cada simulación se desarrolla durante un periodo continuo de $168$ horas ($7$ días completos). Estas configuraciones, junto con otras relacionadas con el sensor de glucosa y la bomba de infusión de insulina, se encuentran en el archivo `DataSimulator.ipynb`. Asimismo, el directorio `data` contiene toda la información y datos generados por el simulador, empleados en el desarrrollo del presente TFG.

---

## Modelo híbrido

El núcleo del proyecto es la implementación de un **Modelo Mínimo de Bergman híbrido**. A diferencia del modelo tradicional de parámetros constantes, esta arquitectura utiliza una red neuronal integrada en el sistema de ecuaciones diferenciales para estimar dinámicamente el parámetro $p_3$.

El sistema de ecuaciones diferenciales ordinarias que rige el modelo es [^1]:

$$
\frac{dG}{dt} = -p_1 \ \big(G(t) - G_b\big) - X(t) \ G(t) + u_G(t)
$$

$$
\frac{dX}{dt} = -p_2 \ X(t) + p_3(t) \ \big(I(t) - I_b\big)
$$

$$
\frac{dI}{dt} = -n \ \big(I(t) - I_b\big) + u_I(t)
$$

Donde la red neuronal se integra directamente en el flujo del integrador numérico para corregir la trayectoria en tiempo real basándose en los datos del paciente.

---

## Preprocesamiento de los datos

Antes de comenzar con el entrenamiento del modelo, se implementa una fase exhaustiva de preprocesamiento de los datos, que consta de los siguientes pasos fundamentales:

- Uso del **filtro de Kalman** para reducir el ruido propio de los sensores CGM, que presentan un retraso fisiológico de aproximadamente $10$ o $15$ minutos, debido al tiempo que tarda en llegar la glucosa desde la sangre hasta el líquido intersticial del tejido donde se coloca el sensor.
  
- Sincronización de frecuencias de muestreo.
  
- Extracción, mediante simulaciones computacionales de los **procesos metabólicos subyacentes**, de variables clínicas necesarias para las ecuaciones de Bergman. Por ejemplo, se modela el tránsito intestinal y el vaciado del estómago para obtener la tasa de absorción de la glucosa.

---

## Estructura del entrenamiento

La aplicación implementa un ciclo de entrenamiento y validación utilizando principalmente **PyTorch**. En este proceso, se emplean técnicas como:

1. **Normalización:** Este paso es fundamental para garantizar un entrenamiento estable de la red neuronal, pero como se trata de un trabajo del campo de la medicina, se debe respetar también el sistema metabólico. Por un lado, la glucosa continua del sensor CGM se normaliza respecto a la glucosa basal del paciente ($G_b$). Por otro lado, el resto de variables y señales se normalizan respecto al percentil $99$ para evitar la pérdida de información válida.

2. **Regularización:** Se lleva a cabo mediante una función de pérdida que combina el error cuadrático medio entre la glucosa predicha ($\hat{y}$) y la medida por el sensor ($y$), el coeficiente de correlación de Pearson, un término de suavidad y una penalización adicional por errores de predicción en episodios de hiperglucemia o hipoglucemia.
     
3. ***Sliding windows*:** Para conseguir un correcto aprendizaje de la red neuronal, es conveniente dividir la información continua del paciente en fragmentos de longitud fija y más reducida. En este trabajo, el tamaño de la ventana se ha fijado en $200$ pasos temporales, que equivalen a $10$ horas por ventana, teniendo en cuenta la frecuencia de muestreo de $3$ minutos. 

4. **Fase de *warm-up*:** Los primeros $30$ pasos temporales se utilizan para que el sistema se estabilice partiendo de las condiciones iniciales ($G_0, \ X_0, \ I_0$). Durante esta fase, el modelo genera predicciones pero no se calcula la función de pérdida.
   
5. **Fase de entrenamiento / inferencia:** El modelo resuelve las ecuaciones diferenciales ordinarias mediante el método de **Runge-Kutta de orden 4 (RK4)**. Se comparan las trayectorias resultantes con los niveles reales de glucosa para actualizar los pesos de la red mediante *backpropagation*.

<p align="center">
  <img width="746" height="343" alt="EjemploEntrenamiento" src="https://github.com/user-attachments/assets/c96ba9d1-9a95-4fe4-891e-764e0065aed0" />
</p>

--- 

## Evaluación del modelo

Para cuantificar la precisión y el rendimiento de los enfoques híbrido y clásico, se calculan una serie de métricas de forma **global e individual** por paciente, habituales en **problemas de regresión de series temporales continuas**:

* MSE (Error Cuadrático Medio)
  
* RMSE (Raíz del Error Cuadrático Medio)
  
* MAE (Error Absoluto Medio)
  
* MAPE (Error Porcentual Absoluto Medio)
  
* $R^2$ (Coeficiente de determinación)

Sin embargo, en contextos médicos como el presente, estas métricas no resultan suficientes porque no consideran el impacto de los errores en los pacientes. Para solucionar estas limitaciones, es necesario incluir otras formas de evaluación adicionales a las métricas, que permitan comprobar la seguridad de las predicciones en entornos clínicos reales. Las **visualizaciones** empleadas en el trabajo son las siguientes:

* Comparación de la trayectoria de la glucosa real respecto a las trayectorias de la glucosa predicha por ambos enfoques (híbrido y clásico) para cada paciente del conjunto de *test*.
  
* Evolución de la función de pérdida en los conjuntos de entrenamiento y de validación a lo largo de las épocas.
  
* Diagrama de violín para representar la distribución del error de los modelos híbrido y clásico.
  
* Análisis de rejilla de error de Clarke [^2]. Incluye líneas que delimitan varias zonas, con el objetivo de indicar la fiabilidad clínica de las predicciones.
  
* Histograma de frecuencias de errores entre la glucosa real y la glucosa predicha por el modelo híbrido, junto con una curva de densidad de distribución normal.

<table style="width:100%">
  <tr valign="bottom">
    <td align="center" width="65%">
      <img src="https://github.com/user-attachments/assets/654121cb-d0f8-4f2c-9449-0be858abdeed" alt="Evolución temporal">
    </td>
    <td align="center" width="35%">
      <img src="https://github.com/user-attachments/assets/3d14e7b3-1af4-468e-9e01-82a49925d91c" alt="Cuadrícula de Clarke">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Figura 1: Evolución temporal de varios pacientes</b>
    </td>
    <td align="center">
      <b>Figura 2: Análisis de rejilla de error de Clarke</b>
    </td>
  </tr>
</table>

---

## Escenarios planteados

Tras conseguir una configuración en la que el modelo híbrido ya presentaba un correcto funcionamiento y un comportamiento estable, se procedió a realizar una fase de optimización. Para alcanzar el mejor rendimiento posible del modelo híbrido, se propusieron varios escenarios modificando determinados hiperparámetros:

* **Escenario $1$: regularización suave de la tendencia** &rarr; `BergmanHybridModel_Scenario1.ipynb`

  La integración de la red neuronal produce una mejora en todas las métricas globales, consiguiendo una mejor precisión y unos valores predichos más cercanos a los reales. Sin embargo, se puede percibir que no alcanza correctamente los extremos de hiperglucemia e hipoglucemia.

* **Escenario $2$: regularización media-alta de la tendencia** &rarr; `BergmanHybridModel_Scenario2.ipynb`

  Prioriza la forma de la curva para conseguir alcanzar los extremos de hiper e hipoglucemia. No obstante, se evidencia un deterioro de la precisión numérica global, ya que todos los errores del modelo híbrido aumentan, disminuyendo por consiguiente la mejora relativa respecto al modelo clásico.

* **Escenario $3$: regularización de los extremos glucémicos** &rarr; `BergmanHybridModel_Scenario3.ipynb`

  Penaliza al modelo si no se ajusta a las desviaciones producidas por los valores extremos correspondientes a episodios de hiperglucemia e hipoglucemia. La adición del nuevo componente en la función de pérdida produce una mejora en las métricas con respecto al Escenario $1$, consiguiendo reducir el MSE global en un $80.3$%. Además, destaca especialmente el caso de un paciente que obtiene una mejora en términos de MSE del $90,30$%. Por último, el análisis de rejilla de error de Clarke indica que aproximadamente un $70$% de las predicciones se encuentran en las zonas A y B (donde se consideran aceptables) y un $40$% en la zona A (donde se consideran precisas). 

---

## Tecnologías empleadas

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="Matplotlib">
  <img src="https://img.shields.io/badge/simglucose-v0.2.1-blueviolet?style=for-the-badge&logo=github" alt="Simglucose">
</p>

* **Lenguaje de programación:** Python (versión 3.11.14).
* **Entorno de desarrollo:** Jupyter Notebook (versión 7.5.3).
* **Librerías y herramientas:** NumPy, Pandas, Scikit-learn, PyTorch, Torchdiffeq, SciPy, Matplotlib, Seaborn, PyKalman y Simglucose.

---

## Cómo ejecutar el proyecto

1. Clonar este repositorio en una máquina local usando la terminal:
   ```bash
   git clone https://github.com/laura30llorente/modelo_hibrido_Bergman_MINMOD.git
   ```
   *Alternativamente, se puede descargar el código en formato ZIP desde el botón «Code» de GitHub.*

2. En el entorno virtual donde se va a ejecutar el *notebook*, instalar las **dependencias necesarias**:
   * Para los *notebooks* de los tres escenarios planteados para el modelo híbrido:
     ```bash
     pip install tqdm pandas numpy seaborn matplotlib torch torchdiffeq scipy scikit-learn pykalman
     ```
   * Para el *notebook* que ejecuta la simulación necesaria para obtener el conjunto de datos utilizado en el trabajo:
     ```bash
     pip install simglucose
     ```
    
3. Ejecucución de la aplicación:
    * Para realizar el **entrenamiento y validación del modelo híbrido**, ejecutar cualquiera de los tres *notebooks* de escenarios planteados (`BergmanHybridModel_ScenarioN.ipynb`). El programa mostrará la evolución de la *loss* de entrenamiento y validación, generando gráficas comparativas al finalizar cada época. Por último, realizará la evaluación y comparación del modelo híbrido frente al clásico usando las métricas y visualizaciones detalladas.
      
    * Para obtener el conjunto de datos a partir del **simulador simglucose**, ejecutar el *notebook* `DataSimulator.ipynb`. El programa solicitará la configuración deseada para la simulación y generará los archivos CSV con los datos metabólicos de cada paciente, junto con sus correspondientes metadatos.
   
---

### Referencias

[^1]:Bergman, R. N., Ider, Y. Z., Bowden, C. R., & Cobelli, C. (1979). Quantitative estimation of insulin sensitivity. *American Journal of Physiology, 236*(6), E667-E677. https://doi.org/10.1152/ajpendo.1979.236.6.E667

[^2]:Clarke, W. L., Cox, D., Gonder-Frederick, L. A., Carter, W., & Pohl, S. L. (1987). Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose. *Diabetes Care, 10*(5), 622-628. https://doi.org/10.2337/diacare.10.5.622

[^3]:Xie, J. (2018). *Simglucose v0.2.1: A Type 1 Diabetes Simulator* [Software]. Disponible en: https://github.com/jxx123/simglucose
