# Ejercicio Final - Kevin Hansen
## Introducción:
El objetivo de este ejercicio final, es utilizar los datos obtenidos a partir la señal de un electroencefalograma (EEG) producida durante 11 minutos, en los que la persona pasó por distintos estados. Además, se cuenta con un video que graba a esta persona durante ese período de tiempo.  
Los datos, el registro de EEG y el video, están disponibles en el siguiente link:  
https://drive.google.com/file/d/1ByQDK4ZPxbqw7T17k--avTcgSCCzs3vi/view?usp=sharing  
  
Los estados identificados, son los siguientes:  
* Baseline  
* Toser  
* Respirar Hondo  
* Respirar Rápido  
* Calculo Mental  
* Colores Violeta  
* Colores Rojo  
* Sonreir  
* Desagradable  
* Agradable  
* Pestañeos Código  
  
Para el análisis, estaré utilizando el ambiente virtual mne3, que puede encontrarse en el siguiente link:  
https://github.com/faturita/python-scientific  
  
## Análisis del dataset, Limpieza y Creación de Features
Comienzo por importar las librerías que estaré utilizando, levantar el dataset, y visualizar algunas filas para ver que formato tienen.
```
import requests
import io
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

# Cargo el archivo desde Github

url = "https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/eeg.dat"
download = requests.get(url).content

df = pd.read_csv(io.StringIO(download.decode('utf-8')), delimiter=' ', names=['timestamp','counter','eeg','attention','meditation','blinking'])

# Imprimo primeras filas para visualizar formatos
print(df.head())
```  
Output:  
```
      timestamp  counter  eeg  attention  meditation  blinking
0  1.655925e+09       68   32          0           0         0
1  1.655925e+09       69   40          0           0         0
2  1.655925e+09       70   42          0           0         0
3  1.655925e+09       71   34          0           0         0
4  1.655925e+09       72   24          0           0         0
```  
Al observar el formato timestamp, y para poder agregar la columna de la categoría del estado por el que está pasando la persona, sumo primero una columna con un timestamp "legible". Luego, clasifico a cada segmento de la señal, según lo que puedo observar que está ocurriendo en el video. Por cuestiones de simplicidad, redondeo al segundo más cercano, y agrego una categoría de "Principio" para los primeros segundos del video. Además, consideraré a todos los momentos de corte, donde la persona parpadea continuamente para indicar el cambio de estado, como "Pestañeos". A fines de realizar un análisis más puntual, solamente estaré comparando los segmentos de Baseline vs. Cuenta Mental, pero el código podría reacomodarse fácilmente para que tome más categorías.  
```
# Defino funcion para agregar categoria en base a los tiempos observados
def categorize(row):
    time = row['time']
    if time >= pd.Timestamp('2022-06-22 19:06:04') and time <= pd.Timestamp('2022-06-22 19:06:05'):
        return 'Principio'
    elif time >= pd.Timestamp('2022-06-22 19:06:10') and time <= pd.Timestamp('2022-06-22 19:07:10'):
        return 'Baseline'
    elif time >= pd.Timestamp('2022-06-22 19:07:12') and time <= pd.Timestamp('2022-06-22 19:08:16'):
        return 'Toser'
    elif time >= pd.Timestamp('2022-06-22 19:08:18') and time <= pd.Timestamp('2022-06-22 19:09:16'):
        return 'Respirar Fondo'
    elif time >= pd.Timestamp('2022-06-22 19:09:18') and time <= pd.Timestamp('2022-06-22 19:10:15'):
        return 'Respirar Rapido'
    elif time >= pd.Timestamp('2022-06-22 19:10:17') and time <= pd.Timestamp('2022-06-22 19:11:14'):
        return 'Calculo Mental'
    elif time >= pd.Timestamp('2022-06-22 19:11:18') and time <= pd.Timestamp('2022-06-22 19:12:16'):
        return 'Colores Violeta'
    elif time >= pd.Timestamp('2022-06-22 19:12:20') and time <= pd.Timestamp('2022-06-22 19:13:16'):
        return 'Colores Rojo'
    elif time >= pd.Timestamp('2022-06-22 19:13:17') and time <= pd.Timestamp('2022-06-22 19:14:18'):
        return 'Sonreir'
    elif time >= pd.Timestamp('2022-06-22 19:14:20') and time <= pd.Timestamp('2022-06-22 19:14:41'):
        return 'Desagradable'
    elif time >= pd.Timestamp('2022-06-22 19:14:42'
                              '') and time <= pd.Timestamp('2022-06-22 19:16:16'):
        return 'Agradable'
    elif time >= pd.Timestamp('2022-06-22 19:16:18') and time <= pd.Timestamp('2022-06-22 19:17:04'):
        return 'Pestañeos Codigo'
    else:
        return 'Pestañeos'

# Agrego categoria
df['category'] = df.apply(categorize, axis=1)

# Filtro la categoria que voy a intentar diferenciar del Baseline
categories = ['Baseline', 'Calculo Mental']
df = df[df.category.isin(categories)]
```  
Teniendo ambos estados a analizar filtrados, procedo a visualizar un gráfico simple de la señal, para ver si puedo identificar patrones individuales o diferencias notables entre ambas.  
```
# Grafico Baseline
df1 = df[df['category'] == 'Baseline']
data1 = df1.values
eeg1 = data1[:,2]

plt.plot(eeg1,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Baseline Unprocessed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-600, 600]);
plt.xlim([0,len(eeg1)])
plt.show()

# Grafico Calculo Mental
df2 = df[df['category'] == 'Calculo Mental']
data2 = df2.values
eeg2 = data2[:,2]

plt.plot(eeg2,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Calculo Mental Unprocessed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-500, 1000]);
plt.xlim([0,len(eeg2)])
plt.show()
```  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Baseline_unprocessed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Calculo_mental_unprocessed.png)  
A simple vista, puedo observar que las señales son bastante distintas, teniendo picos de mayores alturas para el caso del cálculo mental, y aparentemente más seguidos. Esto es interesante, dado que por lo general, puedo asociar los picos del Baseline con parpadeos, pero la frecuencia de parpadeos no aumentó mientras se efectuaban los cálculos mentales. Aún así, se hace difícil clasificar punto a punto si corresponde a una u otra categoría.
  
Para intentar lograr una mejor clasificación de la señal, realizo una limpieza utilizando detrend, y luego genero nuevas features basadas en análisis de la frecuencia, utilizando filtros butter_bandpass. Para estas pruebas, estaré filtrando las ondas theta, alpha_low, alpha_high, beta_low, beta_high, gamma_low y gamma_mid. No utilizaré ondas delta, dado que en posteriores análisis, podía observar que para las categorías seleccionadas, estas ondas no se encontraban presentes.  
  
Una vez creadas estas features, voy a estar calculando su promedio y desvío estándar como features adicionales, dado que por lo general, suelen ser mejores variables predictoras que las bandas en sí. Además, utilizo el StandardScaler para normalizarlas, y procedo a filtrar filas que puedan llegar a haber generado valores NaN o infinitos en su creación, para evitar problemas en los entrenamientos.  
```
# Limpiar la señal EEG utilizando la función detrend
df['eeg'] = detrend(df['eeg'])

# Bandas de Frecuencia
Fs = 512
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Agrego features por bandas de frecuencia
# df['delta'] = butter_bandpass_filter(df['eeg'], 0.5, 2.75, Fs, order=5)
df['theta'] = butter_bandpass_filter(df['eeg'], 3.5, 6.75, Fs, order=5)
df['alpha_low'] = butter_bandpass_filter(df['eeg'], 7.5, 9.25, Fs, order=5)
df['alpha_high'] = butter_bandpass_filter(df['eeg'], 10.0, 11.75, Fs, order=5)
df['beta_low'] = butter_bandpass_filter(df['eeg'], 13.0, 16.75, Fs, order=5)
df['beta_high'] = butter_bandpass_filter(df['eeg'], 18.0, 29.75, Fs, order=5)
df['gamma_low'] = butter_bandpass_filter(df['eeg'], 31.0, 39.75, Fs, order=5)
df['gamma_mid'] = butter_bandpass_filter(df['eeg'], 41.0, 49.75, Fs, order=5)

# Creación de nuevas features
# df['delta_mean'] = df.groupby(['timestamp'])['delta'].transform('mean')
# df['delta_std'] = df.groupby(['timestamp'])['delta'].transform('std')
df['theta_mean'] = df.groupby(['timestamp'])['theta'].transform('mean')
df['theta_std'] = df.groupby(['timestamp'])['theta'].transform('std')
df['alpha_low_mean'] = df.groupby(['timestamp'])['alpha_low'].transform('mean')
df['alpha_low_std'] = df.groupby(['timestamp'])['alpha_low'].transform('std')
df['alpha_high_mean'] = df.groupby(['timestamp'])['alpha_high'].transform('mean')
df['alpha_high_std'] = df.groupby(['timestamp'])['alpha_high'].transform('std')
df['beta_low_mean'] = df.groupby(['timestamp'])['beta_low'].transform('mean')
df['beta_low_std'] = df.groupby(['timestamp'])['beta_low'].transform('std')
df['beta_high_mean'] = df.groupby(['timestamp'])['beta_high'].transform('mean')
df['beta_high_std'] = df.groupby(['timestamp'])['beta_high'].transform('std')
df['gamma_low_mean'] = df.groupby(['timestamp'])['gamma_low'].transform('mean')
df['gamma_low_std'] = df.groupby(['timestamp'])['gamma_low'].transform('std')
df['gamma_mid_mean'] = df.groupby(['timestamp'])['gamma_mid'].transform('mean')
df['gamma_mid_std'] = df.groupby(['timestamp'])['gamma_mid'].transform('std')

# Normalización de features
scaler = StandardScaler()
df[['eeg', 'theta_mean', 'theta_std', 'alpha_low_mean', 'alpha_low_std', 'alpha_high_mean',
    'alpha_high_std', 'beta_low_mean', 'beta_low_std', 'gamma_low_mean', 'gamma_low_std', 'gamma_mid_mean',
    'gamma_mid_std']] = scaler.fit_transform(df[['eeg', 'theta_mean', 'theta_std',
                                                 'alpha_low_mean', 'alpha_low_std', 'alpha_high_mean', 'alpha_high_std',
                                                 'beta_low_mean', 'beta_low_std', 'gamma_low_mean', 'gamma_low_std',
                                                 'gamma_mid_mean', 'gamma_mid_std']])

# Elimino NaNs y valores infinitos
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
```  
Como sé previamente que al realizar cálculos mentales, uno suele manifestar cambios en las ondas beta y gamma, procedo a ver sus gráficos, para intentar entender si estas variables serán mejores predictoras que las originales.
```
# Grafico Baseline Beta Low Processed
df1 = df[df['category'] == 'Baseline'][['timestamp', 'beta_low_mean']]
data1 = df1.values
eeg1 = data1[:,1]

plt.plot(eeg1,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Baseline Beta Low Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-8, 8]);
plt.xlim([0,len(eeg1)])
plt.show()

# Grafico Calculo Mental Beta Low Processed
df2 = df[df['category'] == 'Calculo Mental'][['timestamp', 'beta_low_mean']]
data2 = df2.values
eeg2 = data2[:,1]

plt.plot(eeg2,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Calculo Mental Beta Low Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-8, 8]);
plt.xlim([0,len(eeg2)])
plt.show()

# Grafico Baseline Beta High Processed
df1 = df[df['category'] == 'Baseline'][['timestamp', 'beta_high_mean']]
data1 = df1.values
eeg1 = data1[:,1]

plt.plot(eeg1,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Baseline Beta High Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-30, 30]);
plt.xlim([0,len(eeg1)])
plt.show()

# Grafico Calculo Mental Beta High Processed
df2 = df[df['category'] == 'Calculo Mental'][['timestamp', 'beta_high_mean']]
data2 = df2.values
eeg2 = data2[:,1]

plt.plot(eeg2,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Calculo Mental Beta High Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-40, 40]);
plt.xlim([0,len(eeg2)])
plt.show()

# Grafico Baseline Gamma Low Processed
df1 = df[df['category'] == 'Baseline'][['timestamp', 'gamma_low_mean']]
data1 = df1.values
eeg1 = data1[:,1]

plt.plot(eeg1,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Baseline Gamma Low Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-8, 8]);
plt.xlim([0,len(eeg1)])
plt.show()

# Grafico Calculo Mental Gamma Low Processed
df2 = df[df['category'] == 'Calculo Mental'][['timestamp', 'gamma_low_mean']]
data2 = df2.values
eeg2 = data2[:,1]

plt.plot(eeg2,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Calculo Mental Gamma Low Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-15, 15]);
plt.xlim([0,len(eeg2)])
plt.show()

# Grafico Baseline Gamma Mid Processed
df1 = df[df['category'] == 'Baseline'][['timestamp', 'gamma_mid_mean']]
data1 = df1.values
eeg1 = data1[:,1]

plt.plot(eeg1,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Baseline Gamma Mid Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-15, 15]);
plt.xlim([0,len(eeg1)])
plt.show()

# Grafico Calculo Mental Gamma Mid Processed
df2 = df[df['category'] == 'Calculo Mental'][['timestamp', 'gamma_mid_mean']]
data2 = df2.values
eeg2 = data2[:,1]

plt.plot(eeg2,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Calculo Mental Gamma Mid Processed')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-30, 20]);
plt.xlim([0,len(eeg2)])
plt.show()
```  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Baseline_beta_low_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Calculo_mental_beta_low_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Baseline_beta_high_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Calculo_mental_beta_high_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Baseline_gamma_low_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Calculo_mental_gamma_low_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Baseline_gamma_mid_processed.png)  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/Calculo_mental_gamma_mid_processed.png)  
  
Como puede observarse, con los filtros realizados ahora sí hemos obtenido señales claramente distintas, por lo que se incluirán todas las features creadas en los modelos.  
  
## Entrenamiento de Modelos y Comparación  
Aunque existen infinidades de modelos distintos que pueden entrenarse con estos datos, se van a comparar tres instancias distintas de Support Vector Machine (lineal, polinomial, y radial), un Random Forest, y una red neuronal. Ninguno de los modelos ha tenido optimización de sus hiperparámetros, dado que lo que se busca con este análisis es la factibilidad de su uso a nivel general. Para poder entrenar los modelos, se aplica LabelEncoder, que pasa la columna "category" para que tome valor 0 en caso de ser el Baseline, o 1 en caso de ser el Calculo Mental. En todos los casos, se utiliza como semilla el 42, y se deja un 20% de la muestra para testear. Para el Random Forest se usaron 100 estimadores, y para la Red Neuronal 600 iteraciones, con el fin de reducir los tiempos de entrenamiento pero obtener resultados adecuados.
```
# Selección de features y split de datos
X = df[['eeg', 'theta_mean', 'theta_std', 'alpha_low_mean', 'alpha_low_std',
        'alpha_high_mean', 'alpha_high_std', 'beta_low_mean', 'beta_low_std', 'gamma_low_mean', 'gamma_low_std',
        'gamma_mid_mean', 'gamma_mid_std']]
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Codificar las etiquetas de la variable objetivo para que sean 1 y 0
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Entrenamiento de modelos
svm_lin = SVC(kernel='linear', random_state=42)
svm_pol = SVC(kernel='poly', random_state=42)
svm_rbf = SVC(kernel='rbf', random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
nn = MLPClassifier(max_iter=600, random_state=42)

svm_lin.fit(X_train, y_train)
svm_pol.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)
rf.fit(X_train, y_train)
nn.fit(X_train, y_train)
```  
Para evaluar a los modelos, se comparan los siguientes elementos:  
* Accuracy  
* Matrices de Confusión  
* Curvas ROC  
* F1 Score  
  
A continuación, se muestra el código utilizado, junto con los resultados obtenidos para cada uno.  
Código:  
```
# Evaluación de modelos
y_pred_svm_lin = svm_lin.predict(X_test)
y_pred_svm_pol = svm_pol.predict(X_test)
y_pred_svm_rbf = svm_rbf.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_nn = nn.predict(X_test)

print("Accuracy SVM_LIN:", accuracy_score(y_test, y_pred_svm_lin))
print("Accuracy SVM_POL:", accuracy_score(y_test, y_pred_svm_pol))
print("Accuracy SVM_RBF:", accuracy_score(y_test, y_pred_svm_rbf))
print("Accuracy RF:", accuracy_score(y_test, y_pred_rf))
print("Accuracy NN:", accuracy_score(y_test, y_pred_nn))

# Matrices de confusión
cm_svm_lin = confusion_matrix(y_test, y_pred_svm_lin)
cm_svm_pol = confusion_matrix(y_test, y_pred_svm_pol)
cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_nn = confusion_matrix(y_test, y_pred_nn)

print("Confusion matrix SVM_LIN:\n", cm_svm_lin)
print("Confusion matrix SVM_POL:\n", cm_svm_pol)
print("Confusion matrix SVM_RBF:\n", cm_svm_rbf)
print("Confusion matrix RF:\n", cm_rf)
print("Confusion matrix NN:\n", cm_nn)

# Curvas ROC
fpr_svm_lin, tpr_svm_lin, thresholds_svm_lin = roc_curve(y_test, y_pred_svm_lin)
fpr_svm_pol, tpr_svm_pol, thresholds_svm_pol = roc_curve(y_test, y_pred_svm_pol)
fpr_svm_rbf, tpr_svm_rbf, thresholds_svm_rbf = roc_curve(y_test, y_pred_svm_rbf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_nn)

roc_auc_svm_lin = auc(fpr_svm_lin, tpr_svm_lin)
roc_auc_svm_pol = auc(fpr_svm_pol, tpr_svm_pol)
roc_auc_svm_rbf = auc(fpr_svm_rbf, tpr_svm_rbf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_nn = auc(fpr_nn, tpr_nn)

plt.figure()
plt.plot(fpr_svm_lin, tpr_svm_lin, color='red', label='SVM_LIN (AUC = %0.2f)' % roc_auc_svm_lin)
plt.plot(fpr_svm_pol, tpr_svm_pol, color='darkorange', label='SVM_POL (AUC = %0.2f)' % roc_auc_svm_pol)
plt.plot(fpr_svm_rbf, tpr_svm_rbf, color='yellow', label='SVM_RBF (AUC = %0.2f)' % roc_auc_svm_rbf)
plt.plot(fpr_rf, tpr_rf, color='green', label='RF (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_nn, tpr_nn, color='blue', label='NN (AUC = %0.2f)' % roc_auc_nn)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# F1 score
f1_svm_lin = f1_score(y_test, y_pred_svm_lin)
f1_svm_pol = f1_score(y_test, y_pred_svm_pol)
f1_svm_rbf = f1_score(y_test, y_pred_svm_rbf)
f1_rf = f1_score(y_test, y_pred_rf)
f1_nn = f1_score(y_test, y_pred_nn)

print("F1 score SVM_LIN:", f1_svm_lin)
print("F1 score SVM_POL:", f1_svm_pol)
print("F1 score SVM_RBF:", f1_svm_rbf)
print("F1 score SVM_RBF:", f1_rf)
print("F1 score SVM_RBF:", f1_nn)
```  
Accuracy:
```
Accuracy SVM_LIN: 0.7303819588915703
Accuracy SVM_POL: 0.7392027960389448
Accuracy SVM_RBF: 0.8251643505034535
Accuracy RF: 0.9996671382208538
Accuracy NN: 0.9226096363485062
```  
Matrices de Confusión:
```
Confusion matrix SVM_LIN:
 [[4899 1261]
 [1979 3878]]
Confusion matrix SVM_POL:
 [[5731  429]
 [2705 3152]]
Confusion matrix SVM_RBF:
 [[5577  583]
 [1518 4339]]
Confusion matrix RF:
 [[6159    1]
 [   3 5854]]
Confusion matrix NN:
 [[5868  292]
 [ 638 5219]]
```  
Curvas ROC:  
![alt text](https://raw.githubusercontent.com/KevinHansen90/itba-ecd-ramele/main/data/ROC_curve.png)  
F1 Score:
```
F1 score SVM_LIN: 0.7053473990542015
F1 score SVM_POL: 0.6679381224835771
F1 score SVM_RBF: 0.8050839595509787
F1 score SVM_RBF: 0.9996584699453552
F1 score SVM_RBF: 0.9181914144968332
```  
  
Comparando los modelos con los resultados obtenidos, tenemos como ganador al Random Forest, que casi a la perfección clasificó ambos casos, con un Accuracy y F1 Score muy cercano a 100%.  
En segundo lugar, tenemos a la Red Neuronal, con ambos valores cerca al 90%. En este caso, los errores cometidos fueron mayormente clasificando incorrectamente algunos puntos del Calculo Mental como Baseline, como puede observarse en la matriz de confusión.  
Por último, encontramos a las Support Vector Machines, con Accuracy y F1 Score que varían entre 70% y 80%. Para los tres casos, ocurre lo mismo que con la Red Neuronal, pero la cantidad de errores cometidos son mayores.  
  
Es importante aclarar, que para seguir profundizando el análisis, podrían buscar mejorarse los siguientes puntos:  
* Clasificar correctamente cada segmento de "category", sin redondear al segundo  
* Agregar otros filtros de limpieza de la señal que puedan eliminar aún más ruido de la misma  
* Crear otras features que tengan en cuenta las ventanas (epochs) de la señal, o features como PSD, lags, rolling windows, combinaciones lineales entre features existentes
* No descartar las filas que contengan NaN, nulos o infinitos, sino darles un tratamiento
* Utilizar distintas semillas para reducir variabilidad  
* Optimizar los hiperparámetros de los modelos elegidos
* Utilizar otros modelos más avanzados de redes neuronales como LSTM
* Stackear distintos modelos, aprovechando las bondades de cada uno
