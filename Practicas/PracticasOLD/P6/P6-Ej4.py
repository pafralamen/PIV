'''
    @author fjpv0005 - Francisco Javier Peña Vela

    Escriba un programa de umbralización en Python que calcule el umbral para
    cada píxel a partir de un entorno del mismo de tamaño MxN. Se le pasará por
    línea de comandos el nombre de fichero de la imagen original, los valores
    M y N, el método de umbralización (general o de Otsu), y el nombre de
    fichero para grabar la imagen resultante.

'''

# Librerias
import cv2
import numpy as np
import sys
import math

# Constantes
TONOS_GRISES = 256

#############
# Funciones #
#############
def umbraliza(img, umbral):
    # Creamos la imagen que retornaremos
    imgUmbralizada = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # Aplicamos la umbralización
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgUmbralizada[i][j] = 0 if img[i][j] < umbral else 255

    # Retornamos la imagen umbralizada
    return imgUmbralizada

def calculaUmbralGeneral(img):

    # Selecciono un umbral inicial
    umbral = int((TONOS_GRISES - 1)/2) # 127

    while True: # Emulación de do-while
        umbralAnterior = umbral

        # Segmentar la imagen a partir de dicho umbral.
        # - G1 es el área con intensidad menor que el umbral
        # - G2 es el área con intensidad mayor o igual que el umbral
        # - m1 es la intensidad media de G1
        # - m2 es la intensidad media de G2
        m1, m2 = intensidadMedia(img, umbral)

        # Obtenemos el nuevo umbral
        umbral = (m1 + m2) / 2

        if (umbral == umbralAnterior): # Cuando se estabilice, salimos
            break

    # Retornamos el umbral obtenido
    return umbral


def intensidadMedia(img, umbral):
    # Calculamos la intensidad media de los pixeles de
    #   G1 (área con intensidad menor que el umbral), es decir,
    #   los pixeles por debajo del umbral
    acumuladorG1 = 0     # Acumulador del área G1
    numeroPixelesG1 = 0  # Contador de pixeles del área G1
    acumuladorG2 = 0     # Acumulador del área G2
    numeroPixelesG2 = 0  # Contador de pixeles del área G2

    # Calculo los valores de G1 y G2
    for i in range(img.shape[0]): # Alto imagen
        for j in range(img.shape[1]): # Ancho imagen
            if (img[i][j] < umbral):
                acumuladorG1 += img[i][j]
                numeroPixelesG1 += 1
            else:
                acumuladorG2 += img[i][j]
                numeroPixelesG2 += 1

    # Retornamos la media de G1 y G2
    return (acumuladorG1/numeroPixelesG1) if numeroPixelesG1 != 0 else 0, (acumuladorG2/numeroPixelesG2) if numeroPixelesG2 != 0 else 0

def calcularHistograma(img):
    # Vector de 256 posiciones de tipo unit32 inicializado a 0
    vectorHistograma = np.zeros(TONOS_GRISES, np.uint32)

    # Obtengo las dimensiones de la imagen
    dimensiones = img.shape
    alto = dimensiones[0] # Y
    ancho = dimensiones[1] # X

    # Calculo los valores del vector del histograma
    for i in range(alto):
        for j in range(ancho):
            vectorHistograma[img[i][j]] += 1

    return vectorHistograma

def calculaUmbralOtsu(img):

    # Obtenemos el histograma
    histograma = calcularHistograma(img)

    # Probabilidades
    probabilidad = np.zeros(TONOS_GRISES, np.float32)
    for i in range(TONOS_GRISES):
        probabilidad[i] = histograma[i] / (img.shape[0]*img.shape[1])

    # P1(k)
    pk = np.zeros(TONOS_GRISES, np.float32)
    pk[0] = probabilidad[0]
    for i in range(1, TONOS_GRISES):
        pk[i] = pk[i - 1] + probabilidad[i]

    # M(k)
    mk = np.zeros(TONOS_GRISES, np.float32)
    mk[0] = probabilidad[0]
    for i in range(1, TONOS_GRISES):
        mk[i] = i * probabilidad[i] + mk[i - 1]

    # M_g
    mg = 0
    for i in range(TONOS_GRISES):
        mg += i * probabilidad[i]

    # Obtenemos el umbral de otsu
    valorMaximo = -1
    umbral = 0
    for i in range(TONOS_GRISES):
        numerador = (((mg * pk[i]) - mk[i])**2)
        denominador = (pk[i] * (1 - pk[i]))
        valorOtsu = numerador / denominador if denominador != 0 else 0
        if (valorOtsu > valorMaximo):
            valorMaximo = valorOtsu
            umbral = i

    # Retornamos el umbral obtenido
    return umbral

'''
    Calcula el entorno con medidas 'altoEntorno' x 'anchoEntorno' de las
    coordenadas 'x' e 'y' en la imagen 'img'

    NOTA --> Esta función ya crea un entorno.

    @param img Imagen de donde se extraerá el entorno
'''

def getEntorno(img, altoEntorno, anchoEntorno, y, x):
    # Declaro e inicializo el entorno
    entorno = np.zeros((altoEntorno, anchoEntorno), np.uint8)

    # Obtengo los valores para recorrer el entorno desde -mitadEntorno hasta +mitadEntorno
    # Si por ejemplo el ancho de entorno es 5, la mitad (redondeando hacia abajo) valdrá 2
    # Y podremos recorrer en entorno de la siguiente manera: -2 -1 0 1 2
    mitadAltoEntorno = math.floor(altoEntorno/2)
    mitadAnchoEntorno = math.floor(anchoEntorno/2)

    for i in range(-1*mitadAltoEntorno, mitadAltoEntorno+1): # +1 para que incluya el último
        for j in range(-1*mitadAnchoEntorno, mitadAnchoEntorno+1): # +1 para que incluya el último

            # Calculo la vertical y horizontal correspondiente en la imagen.
            # Hay que sacar los valores para cambiarlos en caso de que se salgan de la imagen.
            vertical = y + i if y+i > 0 and y+i < img.shape[0] else 0
            horizontal = x + j if x+j > 0 and x+j < img.shape [1] else 0

            """
                i + mitadAltoEntorno --> Para que guarde los valores en el entorno desde 0 hasta altoEntorno
                j + mitadAnchoEntorno --> Para que guarde los valores en el entorno desde 0 hasta anchoEntorno
            """
            entorno[i + mitadAltoEntorno][j + mitadAnchoEntorno] = img[vertical][horizontal]

    return entorno


def umbralizaPorEntorno(img, altoEntorno, anchoEntorno, metodo):

    # Declaro e inicializo con ceros la imagen que retornaremos
    imgUmbralizada = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    contador = img.shape[0] * img.shape[1]

    # Recorro todos los pixeles de la imagen
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            # Obtengo el entorno
            entorno = getEntorno(img, altoEntorno, anchoEntorno, i, j)


            if (metodo == "General"):
                entorno = umbraliza(entorno, calculaUmbralGeneral(entorno))
            if (metodo == "Otsu"):
                entorno = umbraliza(entorno, calculaUmbralOtsu(entorno))

            imgUmbralizada[i][j] = entorno[math.floor(altoEntorno/2)][math.floor(anchoEntorno/2)]
            #print("Quedan ", contador)
            contador-=1
            #if imgUmbralizada[i][j] != 255:
            #print("Valor de la imagen umbralizada en la posición ",i,":",j," --> ",imgUmbralizada[i][j])


    # Retornamos la imagen umbralizada
    return imgUmbralizada



############
# Programa #
############

# TEMP: sys.argv = ["ejercicio4.py","./p6.png", 3, 3, "Otsu", "resultado_ej4.png"]

if (len(sys.argv) < 6):
    print("Falta(n) ", 6 - len(sys.argv)," argumento(s).")
    sys.exit()

# Recupero los argumentos de la linea de comandos
pathImg = sys.argv[1]        # <-- Ruta a la imagen
M = sys.argv[2]              # <-- Alto del entorno
N = sys.argv[3]              # <-- Ancho del entorno
metodo = sys.argv[4]         # <-- Método a utilizar (General u Otsu)
nombreImgFinal = sys.argv[5] # <-- Nombre de la imagen final

imgOriginal = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)
imgUmbralizada = umbralizaPorEntorno(imgOriginal, M, N, metodo)
cv2.imshow("Imagen original umbralizada por entorno", imgUmbralizada)
#cv2.imwrite(nombreImgFinal, imgUmbralizada)
cv2.waitKey(0)
cv2.destroyAllWindows()