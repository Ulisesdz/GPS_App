from grafo import *
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
"""
callejero.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

    Grupo: GP012
    Integrantes:
        - Ulises Díez Santaolalla
        - Ignacion Felices Vera

Descripción:
Librería con herramientas y clases auxiliares necesarias para la representación de un callejero en un grafo.

Complétese esta descripción según las funcionalidades agregadas por el grupo.
"""


#Constantes con las velocidades máximas establecidas por el enunciado para cada tipo de vía.
VELOCIDADES_CALLES={"AUTOVIA":100,"AVENIDA":90,"CARRETERA":70,"CALLEJON":30,"CAMINO":30,"ESTACION DE METRO":20,"PASADIZO":20,"PLAZUELA":20,"COLONIA":20}
VELOCIDAD_CALLES_ESTANDAR=50

def datasets():
    path1 = 'CRUCES.csv' 
    path2 = 'DIRECCIONES.csv'
    dfcruces_processed, dfdirecciones_processed = process_data(path1,path2)

    return dfcruces_processed, dfdirecciones_processed


class Cruce:

    #Completar esta clase con los datos y métodos que se necesite asociar a cada cruce

    def __init__(self,coord_x,coord_y,calles,nombre_via):
        self.coord_x=coord_x
        self.coord_y=coord_y
        self.calles = calles
        self.nombre_via = nombre_via

        #Completar la inicialización de las estructuras de datos agregadas

   
    
    
    """Se hace que la clase Cruce sea "hashable" mediante la implementación de los métodos
    __eq__ y __hash__, haciendo que dos objetos de tipo Cruce se consideren iguales cuando
    sus coordenadas coincidan (es decir, C1==C2 si y sólo si C1 y C2 tienen las mismas coordenadas),
    independientemente de los otros campos que puedan estar almacenados en los objetos.
    La función __hash__ se adapta en consecuencia para que sólo dependa del par (coord_x, coord_y).
    """
    def __eq__(self,other) -> int:
        if type(other) is type(self):
            return ((self.coord_x==other.coord_x) and (self.coord_y==other.coord_y))
        else:
            return False
    
    def __hash__(self) -> int:
        return hash((self.coord_x,self.coord_y))
    
#Apartado 4


def distancia_entre_puntos(p1, p2):
    import numpy as np
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def unificar(r,dfcruces_processed):
    dfcruces_processed['Coordenadas X, Y'] = dfcruces_processed.apply(lambda row: (row['Coordenada X (Guia Urbana) cm (cruce)'], row['Coordenada Y (Guia Urbana) cm (cruce)']), axis=1)
    coordenadas = dfcruces_processed['Coordenadas X, Y'].unique()
    #clave lista ordenada para iterar rapido
    sorted_coordinates = sorted(coordenadas)
    grupos = []
    
    i = 0
    while i < len(sorted_coordinates) - 1:
        grupo = sorted_coordinates[i]
        j = i + 1
        nuevo_contador = False
        #En este bucle, al tener la lista ordenada itera las siguientes tuplas para observar si es menor que el radio 
        #nuevo indice donde empezar, el primero que se salga
        #los que si que estan los quitas
        # iterar para (x-coordinate) + r 
        while j < len(sorted_coordinates) and (j <= i + r):
            if nuevo_contador == False and distancia_entre_puntos(sorted_coordinates[i], sorted_coordinates[j]) > r:
                i = j
                nuevo_contador = True
            elif distancia_entre_puntos(sorted_coordinates[i], sorted_coordinates[j]) < r:
                sorted_coordinates.remove(sorted_coordinates[j])
                j -= 1

            j += 1
    
    
        grupos.append(grupo)
        
    
    return grupos

def unificar_1(r,dfcruces_processed):
    import numpy as np
    from scipy.spatial import cKDTree
    dfcruces_processed['Coordenadas X, Y'] = dfcruces_processed.apply(lambda row: (row['Coordenada X (Guia Urbana) cm (cruce)'], row['Coordenada Y (Guia Urbana) cm (cruce)']), axis=1)

    #clave lista ordenada para iterar rapido
    coordenadas = np.array(dfcruces_processed['Coordenadas X, Y'].unique().tolist())
    
    # Build KD-tree for efficient spatial searches
    tree = cKDTree(coordenadas)
    grupos = {}
    nodos = []
    for coord in coordenadas:
        # Hace una lista con los indices de la lista coordenadas cuyas cordnadas (x,y) se encuentrar dentro de una distancia 'r'
        indices = tree.query_ball_point(coord, r)
        indices.remove(indices[0])
        key = tuple(coord)
        if indices != []:
            grupos[key] = []
        for index in indices:
            punto = tuple(coordenadas[index])
            if distancia_entre_puntos(coord, punto) < r:
                grupos[key].append(punto)
        

    return grupos

#Apartado 6
def cruces(dfcruces_processed):

    #Cogemos las calles principales que son cruzadas
    dfcruces_processed['Coordenadas X, Y'] = dfcruces_processed.apply(lambda row: (row['Coordenada X (Guia Urbana) cm (cruce)'], row['Coordenada Y (Guia Urbana) cm (cruce)']), axis=1)
    cruces_unicos = dfcruces_processed['Coordenadas X, Y'].unique()

    cruces_maps = []
    for cruce in cruces_unicos:
            
        filtered_rows = dfcruces_processed[dfcruces_processed['Coordenadas X, Y'] == cruce]
        lista_calles = filtered_rows[['Codigo de via que cruza o enlaza', 'Codigo de vía tratado']].values.flatten()
        calles = list(set(lista_calles))
        nombre_via = filtered_rows['Nombre de la via tratado'].iloc[0]
        #meto las cordenadas-x,cordenadas-y, y una lista de calles del cruce (sus codigos) 
        cruce = Cruce(cruce[0], cruce[1], calles, nombre_via)        
        cruces_maps.append(cruce)

    return cruces_maps

def cruces_TODOS(dfcruces_processed):

    #Hacemos esta funcion para que a la hora de hacer las instrucciones, tengamos TODOS los cruces de Madird ya que nos interesa
    #tenerlos todos, para tener para cada uno el vial cuando es le principal o cuando es el que cruza
    dfcruces_processed['Coordenadas X, Y'] = dfcruces_processed.apply(lambda row: (row['Coordenada X (Guia Urbana) cm (cruce)'], row['Coordenada Y (Guia Urbana) cm (cruce)']), axis=1)
    cruces_unicos = dfcruces_processed['Coordenadas X, Y'].unique()

    cruces_maps = []
    for cruce in cruces_unicos:
            
        filtered_rows = dfcruces_processed[dfcruces_processed['Coordenadas X, Y'] == cruce]
        
        lista_calles = filtered_rows[['Codigo de via que cruza o enlaza', 'Codigo de vía tratado']].values.flatten()
        calles = list(set(lista_calles))
        for _,row in filtered_rows.iterrows():
            nombre_via = row['Nombre de la via tratado']
            #meto las cordenadas-x,cordenadas-y, y una lista de calles del cruce (sus codigos) 
            cruce_objeto = Cruce(cruce[0], cruce[1], calles, nombre_via)        
            cruces_maps.append(cruce_objeto)

    return cruces_maps


class Calle:
    #Completar esta clase con los datos que sea necesario almacenar de cada calle para poder reconstruir los datos del 
    def __init__(self,codigo_calle,cruces,numero,diccionario_numeros,tipo_de_via, nombre_calle) -> None:
        self.codigo_calle=codigo_calle
        self.cruces=cruces
        self.numero = numero
        self.diccionario_numeros = diccionario_numeros
        self.tipo_de_via = tipo_de_via
        self.nombre_calle = nombre_calle


#Apatado 5 

def calles(dfdirecciones_processed,dfcruces_processed):

    calles_maps = []

    lista_1 = list(dfcruces_processed['Codigo de vía tratado'])
    lista_2 = list(dfcruces_processed['Codigo de via que cruza o enlaza'])
    lista_juntas = lista_1 + lista_2

    codigos_unicos = list(set(lista_juntas))


    for codigo_calle in codigos_unicos:
        
        filtered_rows = dfcruces_processed[dfcruces_processed['Codigo de vía tratado'] == codigo_calle]
        cruces = []
        for _, row in filtered_rows.iterrows():
            cruce = (row['Coordenada X (Guia Urbana) cm (cruce)'], row['Coordenada Y (Guia Urbana) cm (cruce)'])
            cruces.append(cruce)


        filtered_rows_1 = dfdirecciones_processed[dfdirecciones_processed['Codigo de via'] == codigo_calle]
        numeros= []
        diccionario = {}
        for _, row in filtered_rows_1.iterrows():
            numero = row['Literal de numeracion']
            numeros.append(numero)
            tipo_de_via = row['Clase de la via']
            nombre_calle = row['Nombre de la vía']
            diccionario[numero]= (row['Coordenada X (Guia Urbana) cm'],row['Coordenada Y (Guia Urbana) cm']) 

            
        
        calle = Calle(codigo_calle, list(set(cruces)), numeros, diccionario, tipo_de_via,nombre_calle)
        calles_maps.append(calle)

    return calles_maps
    
#Apartado 7 Agregar Vertices:
def agregar_vertices(grafo,lista_cruces):
    dicc = {}
    for cruce in lista_cruces:
        grafo.agregar_vertice((cruce.coord_x, cruce.coord_y))
        dicc[(cruce.coord_x,cruce.coord_y)] = (cruce.coord_x,cruce.coord_y)

    return dicc


def agregar_aristas_peso_dist_euc(grafo,lista_calles):
    for calle in lista_calles:
        dict_numeros_calle = calle.diccionario_numeros
        claves = list(dict_numeros_calle.keys())
        if len(dict_numeros_calle) != 0:
            clave_min = min(claves)
            coordenadas_min = dict_numeros_calle[clave_min]
            coordenadas_ref = (coordenadas_min[0],coordenadas_min[1])
            cruces_ordenados = sorted(calle.cruces,key = lambda cruce: distancia_entre_puntos(coordenadas_ref,(cruce[0],cruce[1])))

            for i in range(len(cruces_ordenados) - 1):  #Ya que el ultimo no es necesario
                ini = cruces_ordenados[i]
                fin = cruces_ordenados[i+1]
                distancia = distancia_entre_puntos(ini,fin)
                datos = {'codigo':calle.codigo_calle, 'vertice1':ini , 'vertice2':fin, 'distancia': distancia}
                grafo.agregar_arista(ini,fin,datos,distancia)   #Aqui añadimos el peso
                
        else:
            if len(calle.cruces) > 1:
                coordenadas_min = calle.cruces[0]
                coordenadas_ref = (coordenadas_min[0], coordenadas_min[1])
                cruces_ordenados = sorted(calle.cruces,key = lambda cruce: distancia_entre_puntos(coordenadas_ref,(cruce[0],cruce[1])))
                for i in range(len(cruces_ordenados) - 1):
                    ini = cruces_ordenados[i]
                    fin = cruces_ordenados[i+1]
                    distancia = distancia_entre_puntos(ini,fin)
                    datos = {'codigo':calle.codigo_calle, 'vertice1':ini , 'vertice2':fin, 'distancia': distancia}
                    grafo.agregar_arista(ini,fin,datos,distancia)   #Aqui añadimos el peso
            else:
                pass

#1. Encontrar el tipo de via
#2. Aplicar la vel max
#3. tiempo = distancia / velocidad

def calcular_tiempo(calle,distancia):
    clase_via = calle.tipo_de_via

    if clase_via == 'AUTOVIA':
        vel_max = 100
    elif clase_via == 'AVENIDA':
        vel_max = 90
    elif clase_via == 'CARRETERA':
        vel_max = 70
    elif clase_via == 'CALLEJON' or clase_via == 'CAMINO':
        vel_max = 30
    elif clase_via == 'PASADIZO' or clase_via == 'PLAZUELA' or clase_via == 'COLONIA' or clase_via == 'ESTACION DE METRO':
        vel_max = 20
    else:
        vel_max = 50
    #Hay que transformar la distancia en metros?
    tiempo = distancia / vel_max 

    return tiempo


def agregar_aristas_peso_vel_max(grafo,lista_calles):
    for calle in lista_calles:
        dict_numeros_calle = calle.diccionario_numeros
        claves = list(dict_numeros_calle.keys())
        if len(dict_numeros_calle) != 0:
            clave_min = min(claves)
            coordenadas_min = dict_numeros_calle[clave_min]
            coordenadas_ref = (coordenadas_min[0],coordenadas_min[1])
            cruces_ordenados = sorted(calle.cruces,key = lambda cruce: distancia_entre_puntos(coordenadas_ref,(cruce[0],cruce[1])))

            for i in range(len(cruces_ordenados) - 1):  #Ya que el ultimo no es necesario
                ini = cruces_ordenados[i] 
                fin = cruces_ordenados[i+1]
                distancia = distancia_entre_puntos(ini,fin)
                tiempo = calcular_tiempo(calle,distancia)
                datos = {'codigo':calle.codigo_calle, 'vertice1':ini , 'vertice2':fin, 'distancia': distancia}
                grafo.agregar_arista(ini,fin,datos,tiempo)   #Aqui añadimos el peso
                
        else:
            if len(calle.cruces) > 1:
                coordenadas_min = calle.cruces[0]
                coordenadas_ref = (coordenadas_min[0], coordenadas_min[1])
                cruces_ordenados = sorted(calle.cruces,key = lambda cruce: distancia_entre_puntos(coordenadas_ref,(cruce[0],cruce[1])))

                for i in range(len(cruces_ordenados) - 1):
                    ini = cruces_ordenados[i]
                    fin = cruces_ordenados[i+1]
                    distancia = distancia_entre_puntos(ini,fin)
                    tiempo = calcular_tiempo(calle,distancia)
                    datos = {'codigo':calle.codigo_calle, 'vertice1':ini , 'vertice2':fin, 'distancia': distancia}
                    grafo.agregar_arista(ini,fin,datos,tiempo)   #Aqui añadimos el peso
            else:
                pass


def generar_Madrid_1(lista_cruces,lista_calles):

    grafo = Grafo()
    dicc = agregar_vertices(grafo,lista_cruces)
    agregar_aristas_peso_dist_euc(grafo,lista_calles)

    grafo_nx = grafo.convertir_a_NetworkX()

    plt.figure(figsize=(50,50))
    plot=plt.plot()
    nx.draw(grafo_nx, pos=dicc, with_labels=False, node_size=0.1)
    nx.draw_networkx_edges(grafo_nx, pos=dicc,  edge_color='b', width=0.4)
    plt.show()

    return grafo


def generar_Madrid_2(lista_cruces,lista_calles):

    grafo = Grafo()
    dicc = agregar_vertices(grafo,lista_cruces)
    agregar_aristas_peso_vel_max(grafo,lista_calles)

    grafo_nx = grafo.convertir_a_NetworkX()

    plt.figure(figsize=(50,50))
    plot=plt.plot()
    nx.draw(grafo_nx, pos=dicc, with_labels=False, node_size=0.1)
    nx.draw_networkx_edges(grafo_nx, pos=dicc,  edge_color='b', width=0.4)
    plt.show()

    return grafo



'''
Estas son las funciones cogidas del dgt_main utilizado para limpiar los datos de la dgt, 
se necesitaban las siguientes funciones para poder procesar los datasets 

ESTAS FUNCIONES NO SON DEL CALLEJERO.PY SI NO DE LA PARACTICA DE ADQUISICION INCLUIDAS SOLO PARA PROCESAR DATASETS!
'''
    

def cruces_read(path_cruces:str):
    #Leo el archivo CSV con la codificación "iso-8859-1" y el separador ";"
    df_cruces = pd.read_csv(path_cruces, encoding='iso-8859-1', sep=";")
    #Devuelvo el DataFrame
    return df_cruces

def clean_names(df_cruces):
    #Selecciono las columnas que voy a corregir
    columnas = [
        'Literal completo del vial tratado',
        'Literal completo del vial que cruza',
        'Clase de la via tratado',
        'Clase de la via que cruza',
        'Particula de la via tratado',
        'Particula de la via que cruza',
        'Nombre de la via tratado',
        'Nombre de la via que cruza'
    ]
    #Elimino los errores de cada columna
    for columna in columnas:
        #Elimino los espacios innecesarios
        df_cruces[columna] = df_cruces[columna].str.strip()
        #Normalizo los errores de codificación
        df_cruces[columna] = df_cruces[columna].str.normalize('NFKD')
    #Devuelvo el DataFrame
    return df_cruces

def cruces_as_int(df_cruces):
    #Selecciono las columnas que voy a convertir en números
    columnas = [
        'Codigo de vía tratado',
        'Codigo de via que cruza o enlaza',
        'Coordenada X (Guia Urbana) cm (cruce)',
        'Coordenada Y (Guia Urbana) cm (cruce)'
    ]
    #Aplico una función lambda para convertir a numero
    funcion_correccion = lambda x: x if isinstance(x, int) else int(x)
    #Aplico la función de conversión a cada columna
    for columna in columnas:
        df_cruces[columna] = df_cruces[columna].apply(funcion_correccion)
    #Devuelvo el DataFrame
    return df_cruces

def direcciones_read(path_direcciones:str):
    #Leo el archivo CSV con la codificación "iso-8859-1" y el separador ";"
    df_direcciones = pd.read_csv(path_direcciones, encoding='iso-8859-1', sep=";")
    #Devuelvo el DataFrame
    return df_direcciones

def direcciones_as_int(df_direcciones):
    #Selecciono las columnas que voy a convertir en números
    columnas = [
        'Codigo de numero',
        'Codigo de via',
        'Coordenada X (Guia Urbana) cm', 
        'Coordenada Y (Guia Urbana) cm'
    ]
    #Aplico una función lambda para convertir a numero y uso Regex por si contiene carácteres no numéricos
    funcion_correccion = lambda x: x if isinstance(x, int) else int(float(re.sub(r'[^0-9.]', '', x)))
    #Aplico la función de conversión a cada columna
    for columna in columnas:
        df_direcciones[columna] = df_direcciones[columna].apply(funcion_correccion)
    #Devuelvo el DataFrame
    return df_direcciones

def literal_split(df_direcciones):
    #Copio el DataFrame para editarlo sin problemas
    df_direcciones_split = df_direcciones.copy()
    #Usando Regex obtengo los tres grupos al separar la cadena de caracteres
    #El primer grupo son letras y puede contener puntos
    #El segundo grupo son números
    #El tercer grupo son letras y puede estar o no
    regex_pattern = r'(^[A-Za-z.]+)([0-9]+)([A-Za-z]+)?'
    #Creo una nueva columna con las coincidencias al aplicar regex del primer grupo
    df_direcciones_split['Prefijo de numeración'] = (df_direcciones_split['Literal de numeracion'].str.findall(regex_pattern)).str[0].str[0]
    #Creo una nueva columna con las coincidencias al aplicar regex del segundo grupo
    df_direcciones_split['Número'] = (df_direcciones_split['Literal de numeracion'].str.findall(regex_pattern)).str[0].str[1]
    #Creo una nueva columna con las coincidencias al aplicar regex del tercer grupo
    df_direcciones_split['Sufijo de numeración'] = (df_direcciones_split['Literal de numeracion'].str.findall(regex_pattern)).str[0].str[2]    
    #Convierto la columna de Numeros en int
    funcion_correccion = lambda x: x if isinstance(x, int) else int(x)
    #Aplico la función de conversión a cada columna
    df_direcciones_split["Número"] = df_direcciones_split["Número"].apply(funcion_correccion)
    #Devuelvo el DataFrame
    return df_direcciones_split

def process_data(path_cruces:str, path_direcciones:str):
    #Obtengo el DataFrame de los cruces
    df_cruces = cruces_read(path_cruces)
    #Limpio los errores del DataFrame de los cruces
    df_cruces_limpio = clean_names(df_cruces)
    #Cambio a numeros las columnas correspondientes
    df_cruces_limpio_int = cruces_as_int(df_cruces_limpio)
    #Obtengo el DataFrame de las direcciones
    df_direcciones = direcciones_read(path_direcciones)
    #Cambio a numeros las columnas correspondientes
    df_direcciones_int = direcciones_as_int(df_direcciones)
    #Creo las tres nuevas columnas en el Dataframe a partir de la columna “Literal de Numeración”
    df_direcciones_int_split = literal_split(df_direcciones_int)
    
    #Devuelvo los dos DataFrames procesados y normalizados
    return df_cruces_limpio_int, df_direcciones_int_split

 
 
    

