# Fase 1 - Proyecto MLOps: Predicción de Obesidad

## 📋 Información del Proyecto

**Equipo**: 52  
**Dataset**: Obesity Estimation Dataset  
**Objetivo**: Desarrollar un modelo de Machine Learning para predecir niveles de obesidad basado en características biométricas y hábitos de vida  
**Fase**: 1 - Análisis Exploratorio y Preparación de Datos  

---

## 🎯 1. Análisis de Requerimientos

### Problemática Identificada
El dataset de obesidad presenta un problema de **clasificación multiclase** donde necesitamos predecir el nivel de obesidad de una persona basado en características biométricas y hábitos de vida. La obesidad es un problema de salud pública crítico que requiere herramientas de predicción precisas para intervención temprana.

### Propuesta de Valor con ML
Una solución de Machine Learning puede:
- **Predecir niveles de obesidad** con alta precisión basado en características medibles
- **Identificar factores de riesgo** más influyentes en el desarrollo de obesidad
- **Facilitar intervenciones tempranas** para prevenir complicaciones de salud
- **Optimizar recursos médicos** dirigiendo atención a pacientes de mayor riesgo
- **Personalizar tratamientos** basado en perfiles de riesgo individuales

### ML Canvas
```
PROBLEMA:
- Clasificación multiclase (7 niveles de obesidad)
- Dataset con 2111 registros y 17 características
- Variables: biométricas, demográficas y hábitos de vida

SOLUCIÓN:
- Modelo de clasificación para predecir niveles de obesidad
- Algoritmos: Random Forest, SVM, Logistic Regression
- Métricas: Accuracy, Precision, Recall, F1-Score

DATOS:
- Fuente: Dataset de obesidad con características biométricas
- Calidad: Limpio después de EDA y preprocesamiento
- Volumen: 2111 registros, adecuado para entrenamiento

IMPACTO:
- Mejora en detección temprana de obesidad
- Optimización de recursos médicos
- Prevención de complicaciones de salud
```

---

## 🔧 2. Manipulación y Preparación de Datos

### Dataset Utilizado
- **Original**: `obesity_estimation_original.csv` (2111, 17) - Referencia limpia
- **Modified**: `obesity_estimation_modified.csv` (2153, 18) - Con problemas para practicar
- **Creado**: Dataset limpio resultado del proceso de limpieza

### Problemas Identificados y Solucionados

#### 🔍 Inconsistencias Detectadas:
1. **Texto inconsistente**: Espacios, mayúsculas/minúsculas mixtas
2. **Valores N/A**: 'N/A', 'unknown', 'bad' como strings
3. **Columna extra**: 'mixed_type_col' no presente en original
4. **Valores extremos**: Edad 706 años, peso 1040 kg, altura 57 metros
5. **Filas extra**: 42 filas adicionales en dataset modificado

#### ✅ Estrategia de Limpieza Implementada:
1. **Eliminación de columna extra**: `mixed_type_col`
2. **Limpieza de texto**: Normalización de espacios y caracteres
3. **Conversión de tipos**: Numéricos y categóricos correctos
4. **Validación de rangos realistas**:
   - Edad: 14-100 años
   - Altura: 1.0-2.5 metros
   - Peso: 20-200 kg
5. **Normalización categórica**: Formato estándar (ej: 'obesity_type_iii')
6. **Imputación inteligente**: Mediana para numéricas, moda para categóricas
7. **Ajuste de dimensiones**: Eliminación de filas extra

### Herramientas Utilizadas
- **Python**: Pandas, NumPy para manipulación de datos
- **Visualización**: Matplotlib, Seaborn para EDA
- **Preprocesamiento**: Scikit-learn para transformaciones
- **Control de versiones**: Git para trazabilidad

---

## 📊 3. Exploración y Preprocesamiento de Datos

### Análisis Exploratorio Realizado

#### 📈 Estadísticas Descriptivas:
- **Dataset final**: (2111, 17) registros y características
- **Valores faltantes**: 0 (después de limpieza)
- **Variables numéricas**: 8 (Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE)
- **Variables categóricas**: 9 (Gender, family_history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, NObeyesdad)

#### 🎯 Variable Objetivo (NObeyesdad):
```
Distribución de clases:
- obesity_type_i: 352 personas (16.7%)
- obesity_type_iii: 310 personas (14.7%)
- obesity_type_ii: 293 personas (13.9%)
- overweight_level_i: 289 personas (13.7%)
- overweight_level_ii: 286 personas (13.5%)
- normal_weight: 283 personas (13.4%)
- insufficient_weight: 271 personas (12.8%)
```

#### 📊 Análisis de Correlaciones:
- **No hay correlaciones fuertes** (|r| > 0.3) entre variables numéricas
- **Variables independientes** entre sí
- **BMI con correlaciones moderadas** con peso y altura (esperado)
- **Favorable para ML**: Sin multicolinealidad severa

### Técnicas de Preprocesamiento Aplicadas

#### 🔄 Transformaciones Realizadas:
1. **Codificación categórica**:
   - Variables binarias: Label encoding (0/1)
   - Variables multiclase: One-hot encoding
2. **Normalización numérica**: StandardScaler aplicado
3. **Eliminación de duplicados**: Datos únicos
4. **Manejo de outliers**: Valores extremos corregidos con rangos realistas

#### 📋 Métricas de Calidad:
- **Completitud**: 100% (sin valores faltantes)
- **Consistencia**: Formato uniforme en todas las variables
- **Validez**: Rangos realistas para variables biométricas
- **Unicidad**: Sin duplicados después de limpieza

---

## 🔄 4. Versionado de Datos

### Estrategia de Versionado Implementada

#### 📁 Estructura de Datos:
```
db/
├── obesity_estimation_original.csv    # Dataset original (referencia)
├── obesity_estimation_modified.csv   # Dataset con problemas
└── obesity_estimation_clean.csv      # Dataset limpio (resultado)
```

#### 🏷️ Versiones Documentadas:
1. **v1.0**: Dataset original (limpio del profesor)
2. **v1.1**: Dataset modificado (con problemas para practicar)
3. **v2.0**: Dataset creado (limpio después de EDA)

#### 📝 Registro de Cambios:
- **Eliminación**: Columna 'mixed_type_col'
- **Corrección**: 165 valores extremos fuera de rangos realistas
- **Imputación**: Valores faltantes con mediana/moda
- **Normalización**: Variables categóricas a formato estándar
- **Ajuste**: Eliminación de 42 filas extra

### Herramientas de Versionado
- **Git**: Control de versiones del código y documentación
- **DVC**: Versionado de datasets (próxima implementación)
- **Documentación**: Registro detallado de modificaciones

---

## 🤖 5. Construcción, Ajuste y Evaluación de Modelos

### Algoritmos Seleccionados

#### 🎯 Justificación de Selección:
Basado en el análisis de correlaciones y características del dataset:

1. **Random Forest**:
   - ✅ Maneja bien variables independientes
   - ✅ Robusto a outliers
   - ✅ Proporciona importancia de características
   - ✅ Bueno para clasificación multiclase

2. **Support Vector Machine (SVM)**:
   - ✅ Efectivo con datos normalizados
   - ✅ Maneja bien espacios de alta dimensión
   - ✅ Bueno para clasificación multiclase

3. **Logistic Regression**:
   - ✅ Interpretable y rápido
   - ✅ Sin problemas de multicolinealidad
   - ✅ Bueno para clasificación multiclase

### Métricas de Evaluación Planificadas

#### 📊 Métricas Principales:
- **Accuracy**: Precisión general del modelo
- **Precision**: Por clase (macro y micro)
- **Recall**: Por clase (macro y micro)
- **F1-Score**: Balance entre precision y recall
- **Confusion Matrix**: Visualización de errores por clase

#### 🎯 Métricas Específicas:
- **Balanced Accuracy**: Considerando desbalance de clases
- **ROC-AUC**: Para evaluación multiclase
- **Cross-validation**: Validación robusta con k-fold

### Estrategia de Entrenamiento

#### 🔄 División de Datos:
- **Train**: 70% (1477 registros)
- **Validation**: 15% (317 registros)
- **Test**: 15% (317 registros)

#### ⚙️ Ajuste de Hiperparámetros:
- **Grid Search**: Búsqueda exhaustiva de parámetros
- **Random Search**: Búsqueda aleatoria para eficiencia
- **Cross-validation**: Validación durante ajuste

---

## 👥 Roles y Responsabilidades del Equipo

### 🧑‍💻 Data Engineer
- **Responsabilidades**:
  - Limpieza y preparación de datos
  - Implementación de pipelines de datos
  - Versionado con DVC
  - Optimización de almacenamiento
- **Actividades Realizadas**:
  - EDA completo del dataset
  - Limpieza de inconsistencias
  - Normalización de variables
  - Documentación de cambios

### 🧑‍🔬 Data Scientist
- **Responsabilidades**:
  - Análisis exploratorio de datos
  - Identificación de patrones y tendencias
  - Selección de características
  - Preprocesamiento avanzado
- **Actividades Realizadas**:
  - Análisis de correlaciones
  - Visualizaciones profesionales
  - Detección de outliers
  - Análisis de distribución de clases

### 🧑‍💻 ML Engineer
- **Responsabilidades**:
  - Construcción de modelos
  - Optimización de hiperparámetros
  - Evaluación de rendimiento
  - Implementación de pipelines ML
- **Actividades Planificadas**:
  - Implementación de algoritmos
  - Validación cruzada
  - Comparación de modelos
  - Optimización de rendimiento

### 🧑‍💼 Product Manager
- **Responsabilidades**:
  - Definición de requerimientos
  - Análisis de valor de negocio
  - Coordinación de equipo
  - Documentación ejecutiva
- **Actividades Realizadas**:
  - Análisis de problemática
  - Propuesta de valor
  - Coordinación de entregables
  - Documentación de resultados

---

## 📈 Resultados Obtenidos

### ✅ Logros de la Fase 1:

#### 🎯 Calidad de Datos:
- **Dataset limpio**: (2111, 17) sin valores faltantes
- **Formato consistente**: Variables normalizadas
- **Rangos válidos**: Valores realistas para variables biométricas
- **Sin duplicados**: Datos únicos y confiables

#### 📊 Insights Clave:
- **Distribución balanceada**: Clases relativamente equilibradas
- **Variables independientes**: Sin multicolinealidad severa
- **BMI relevante**: Correlaciones esperadas con peso/altura
- **Hábitos influyentes**: Variables de estilo de vida importantes

#### 🔧 Preparación para ML:
- **Datos preprocesados**: Listos para entrenamiento
- **Características seleccionadas**: Todas las variables relevantes
- **Formato estándar**: Compatible con algoritmos ML
- **Documentación completa**: Proceso reproducible

---

## 🎯 Conclusiones y Reflexiones

### ✅ Fortalezas del Análisis:
1. **Limpieza exhaustiva**: Identificación y corrección de todos los problemas
2. **Análisis profundo**: EDA completo con visualizaciones profesionales
3. **Documentación detallada**: Proceso completamente documentado
4. **Preparación sólida**: Dataset listo para modelado ML

### 🔄 Áreas de Mejora:
1. **Versionado avanzado**: Implementar DVC para mejor trazabilidad
2. **Análisis de características**: Feature engineering más avanzado
3. **Validación externa**: Comparación con datasets similares
4. **Automatización**: Pipelines más automatizados

### 🚀 Estrategias Implementadas:
1. **Enfoque sistemático**: Proceso estructurado de limpieza
2. **Validación continua**: Comparación con dataset original
3. **Documentación en tiempo real**: Registro de cada cambio
4. **Preparación para escalabilidad**: Estructura preparada para ML

### 📋 Próximos Pasos:
- ✅ Dataset limpio y listo para modelado
- ✅ Análisis exploratorio completado
- ✅ Preprocesamiento aplicado
- ✅ Verificación de calidad realizada
- 🔄 Implementación de modelos ML
- 🔄 Evaluación y comparación de algoritmos
- 🔄 Optimización de hiperparámetros

---

## 📁 Estructura del Proyecto

```
Equipo52_MLOPS/
├── README.md
├── requirements.txt
├── FASE 1 Avance del proyecto/
│   ├── db/
│   │   ├── obesity_estimation_original.csv
│   │   └── obesity_estimation_modified.csv
│   └── notebooks/
│       └── EDA.ipynb
└── docs/
    └── (documentación adicional)
```

---

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Ejecución del EDA
```bash
jupyter notebook FASE\ 1\ Avance\ del\ proyecto/notebooks/EDA.ipynb
```

---

## 📞 Contacto del Equipo

**Equipo 52 - MLOps**  
**Instituto Tecnológico de Monterrey**

---

*Este documento representa el avance de la Fase 1 del proyecto MLOps para predicción de obesidad, demostrando competencias en análisis exploratorio, limpieza de datos y preparación para modelado de Machine Learning.*