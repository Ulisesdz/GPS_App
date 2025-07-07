"""
grafo.py

Integrantes:
    - Ulises Díez Santaolalla
    - Ignacion Felices Vera

Descripción:
Librería para la creación y análisis de grafos dirigidos y no dirigidos.
"""

from typing import List, Tuple, Dict
import networkx as nx
import sys
import heapq

INFTY = sys.float_info.max  # Distincia "infinita" entre nodos de un grafo


class Grafo:
    """
    Clase que almacena un grafo dirigido o no dirigido y proporciona herramientas
    para su análisis.
    """

    def __init__(self, dirigido: bool = False):
        """Crea un grafo dirigido o no dirigido.

        Args:
            dirigido (bool): Flag que indica si el grafo es dirigido (verdadero) o no (falso).

        Returns:
            Grafo o grafo dirigido (según lo indicado por el flag)
            inicializado sin vértices ni aristas.
        """

        # Flag que indica si el grafo es dirigido o no.
        self.dirigido = dirigido

        """
        Diccionario que almacena la lista de adyacencia del grafo.
        adyacencia[u]:  diccionario cuyas claves son la adyacencia de u
        adyacencia[u][v]:   Contenido de la arista (u,v), es decir, par (a,w) formado
                            por el objeto almacenado en la arista "a" (object) y su peso "w" (float).
        """
        self.adyacencia: Dict[object, Dict[object, Tuple[object, float]]] = {}

    def hasheable(self, objeto):
        # Esta función intenta devolver el hash del objeto. Si no es posible, lanza una excepción.
        try:
            return hash(objeto)
        # Si no es hasheable, el except recoge el TypeError de la funcion hash e imprime un mensaje, siempre devolviendo un None
        except TypeError:
            print(f"El objeto {objeto} no es hasheable")
            return None

    #### Operaciones básicas del TAD ####
    def es_dirigido(self) -> bool:
        """Indica si el grafo es dirigido o no

        Args: None
        Returns: True si el grafo es dirigido, False si no.
        Raises: None
        """
        if self.dirigido == False:
            return False
        else:
            return True

    def agregar_vertice(self, v: object) -> None:
        """Agrega el vértice v al grafo.

        Args:
            v (object): vértice que se quiere agregar. Debe ser "hashable".
        Returns: None
        Raises:
            TypeError: Si el objeto no es "hashable".
        """
        # Para cada vertice se crea un diccionario vacio
        if self.hasheable(v):
            self.adyacencia[v] = {}

    def agregar_arista(
        self, s: object, t: object, data: object = None, weight: float = 1
    ) -> None:
        """Si los objetos s y t son vértices del grafo, agrega
        una arista al grafo que va desde el vértice s hasta el vértice t
        y le asocia los datos "data" y el peso weight.
        En caso contrario, no hace nada.

        Args:
            s (object): vértice de origen (source).
            t (object): vértice de destino (target).
            data (object, opcional): datos de la arista. Por defecto, None.
            weight (float, opcional): peso de la arista. Por defecto, 1.
        Returns: None
        Raises:
            TypeError: Si s o t no son "hashable".
        """

        if self.hasheable(s) and self.hasheable(t):
            if s in self.adyacencia and t in self.adyacencia:
                self.adyacencia[s][t] = (data, weight)

                # Si es un grafo no dirigido, las aristas han de implementarse en los dos sentidos
                if not self.dirigido:
                    self.adyacencia[t][s] = (data, weight)

    def eliminar_vertice(self, v: object) -> None:
        """Si el objeto v es un vértice del grafo lo elimiina.
        Si no, no hace nada.

        Args:
            v (object): vértices que se quiere eliminar.
        Returns: None
        Raises:
            TypeError: Si v no es "hashable".
        """

        if self.hasheable(v):
            if v in self.adyacencia:
                del self.adyacencia[v]

                for u in self.adyacencia:
                    if v in self.adyacencia[u]:
                        del self.adyacencia[u][v]

    def eliminar_arista(self, s: object, t: object) -> None:
        """Si los objetos s y t son vértices del grafo y existe
        una arista de s a t la elimina.
        Si no, no hace nada.

        Args:
            s: vértice de origen de la arista (source).
            t: vértice de destino de la arista (target).
        Returns: None
        Raises:
            TypeError: Si s o t no son "hashable".
        """

        if self.hasheable(s) and self.hasheable(t):
            if (
                (s in self.adyacencia)
                and (t in self.adyacencia)
                and (t in self.adyacencia[s])
            ):
                del self.adyacencia[s][t]

                # Al ser no dirigido, hay que quitar las aristas en los dos sentidos
                if not self.dirigido:
                    del self.adyacencia[t][s]

    def obtener_arista(self, s: object, t: object) -> Tuple[object, float] or None:
        """Si los objetos s y t son vértices del grafo y existe
        una arista de u a v, devuelve sus datos y su peso en una tupla.
        Si no, devuelve None

        Args:
            s: vértice de origen de la arista (source).
            t: vértice de destino de la arista (target).
        Returns:
            Tuple[object,float]: Una tupla (a,w) con los datos "a" de la arista (s,t) y su peso
                "w" si la arista existe.
            None: Si la arista (s,t) no existe en el grafo.
        Raises:
            TypeError: Si s o t no son "hashable".
        """

        if self.hasheable(s) and self.hasheable(t):
            if (s in self.adyacencia) and (t in self.adyacencia[s]):
                return self.adyacencia[s][t]
            else:
                return None

    def lista_vertices(self) -> List[object]:
        """Devuelve una lista con los vértices del grafo.

        Args: None
        Returns:
            List[object]: Una lista [v1,v2,...,vn] de los vértices del grafo.
        Raises: None
        """

        # list te coge los keys de forma automatica
        return list(self.adyacencia)

    def lista_adyacencia(self, u: object) -> List[object] or None:
        """Si el objeto u es un vértice del grafo, devuelve
        su lista de adyacencia, es decir, una lista [v1,...,vn] con los vértices
        tales que (u,v1), (u,v2),..., (u,vn) son aristas del grafo.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns:
            List[object]: Una lista [v1,v2,...,vn] de los vértices del grafo
                adyacentes a u si u es un vértice del grafo
            None: si u no es un vértice del grafo
        Raises:
            TypeError: Si u no es "hashable".
        """

        if self.hasheable(u) and (u in self.adyacencia):
            lista_adyaciencia = []
            for vertice_adyaciente in self.adyacencia[u].keys():
                lista_adyaciencia.append(vertice_adyaciente)
            return lista_adyaciencia
        else:
            return None

    #### Grados de vértices ####
    def grado_saliente(self, v: object) -> int or None:
        """Si el objeto v es un vértice del grafo, devuelve
        su grado saliente, es decir, el número de aristas que parten de v.
        Si no, devuelve None.

        Args:
            v (object): vértice del grafo
        Returns:
            int: El grado saliente de u si el vértice existe
            None: Si el vértice no existe.
        Raises:
            TypeError: Si u no es "hashable".
        """

        if self.hasheable(v) and (v in self.adyacencia):
            return len(self.adyacencia[v].keys())
        else:
            return None

    def grado_entrante(self, v: object) -> int or None:
        """Si el objeto v es un vértice del grafo, devuelve
        su grado entrante, es decir, el número de aristas que llegan a v.
        Si no, devuelve None.

        Args:
            v (object): vértice del grafo
        Returns:
            int: El grado entrante de u si el vértice existe
            None: Si el vértice no existe.
        Raises:
            TypeError: Si v no es "hashable".
        """

        if self.hasheable(v) and (v in self.adyacencia):
            contador = 0

            for vertice in self.adyacencia.keys():
                if vertice != v:
                    for vertices_adyacientes in self.adyacencia[vertice].keys():
                        if vertices_adyacientes == v:
                            contador += 1
            return contador

        else:
            return None

    def grado(self, v: object) -> int or None:
        """Si el objeto v es un vértice del grafo, devuelve
        su grado si el grafo no es dirigido y su grado saliente si
        es dirigido.
        Si no pertenece al grafo, devuelve None.

        Args:
            v (object): vértice del grafo
        Returns:
            int: El grado grado o grado saliente de u según corresponda
                si el vértice existe
            None: Si el vértice no existe.
        Raises:
            TypeError: Si v no es "hashable".
        """
        # Ya que para los dos tipos de grafos, el grado devuelve el equivalente al saliente
        if self.hasheable(v) and (v in self.adyacencia):
            return self.grado_saliente(v)
        else:
            return None

    #### Algoritmos ####

    def dijkstra(self, origen: object) -> Dict[object, object]:
        """Calcula un Árbol de Caminos Mínimos para el grafo partiendo
        del vértice "origen" usando el algoritmo de Dijkstra. Calcula únicamente
        el árbol de la componente conexa que contiene a "origen".

        Args:
            origen (object): vértice del grafo de origen
        Returns:
            Dict[object,object]: Devuelve un diccionario que indica, para cada vértice alcanzable
                desde "origen", qué vértice es su padre en el árbol de caminos mínimos.
        Raises:
            TypeError: Si origen no es "hashable".
        Example:
            Si G.dijksra(1)={2:1, 3:2, 4:1} entonces 1 es padre de 2 y de 4 y 2 es padre de 3.
            En particular, un camino mínimo desde 1 hasta 3 sería 1->2->3.
        """
        if self.hasheable(origen):
            # Inicializa las distancias y padres de los vértices
            distancias = {vertice: float("inf") for vertice in self.adyacencia}
            padres = {vertice: None for vertice in self.adyacencia}
            # La distancia desde el origen hasta él mismo es 0
            distancias[origen] = 0

            # Inicializa el heap con la tupla (distancia, vértice)
            heap = [(0, origen)]
            # Conjunto de vértices no visitados
            vertices_restantes = set(self.adyacencia.keys())

            # Mientras haya vértices no visitados
            while vertices_restantes and heap:
                # Extrae el vértice con la menor distancia actual desde el heap
                distancia_actual, vertice_actual = heapq.heappop(heap)
                if vertice_actual in vertices_restantes:
                    vertices_restantes.remove(vertice_actual)

                # Itera sobre los vecinos del vértice actual
                for vecino, peso in self.adyacencia[vertice_actual].items():
                    # Calcula la nueva distancia desde el origen hasta el vecino
                    nueva_distancia = distancias[vertice_actual] + peso[1]
                    # Si la nueva distancia es menor que la almacenada, actualiza la distancia y el padre
                    if nueva_distancia < distancias[vecino]:
                        distancias[vecino] = nueva_distancia
                        padres[vecino] = vertice_actual
                        # Agrega el vecino al heap con su nueva distancia
                        heapq.heappush(heap, (nueva_distancia, vecino))

            # Como no nos interesa que salge el vertice None lo borramos
            for nodo in list(padres.keys()):
                if padres[nodo] == None:
                    del padres[nodo]

            return padres

    def camino_minimo(self, origen: object, destino: object) -> List[object]:
        """Calcula el camino mínimo desde el vértice origen hasta el vértice
        destino utilizando el algoritmo de Dijkstra.

        Args:
            origen (object): vértice del grafo de origen
            destino (object): vértice del grafo de destino
        Returns:
            List[object]: Devuelve una lista con los vértices del grafo por los que pasa
                el camino más corto entre el origen y el destino. El primer elemento de
                la lista es origen y el último destino.
        Example:
            Si G.dijksra(1,4)=[1,5,2,4] entonces el camino más corto en G entre 1 y 4 es 1->5->2->4.
        Raises:
            TypeError: Si origen o destino no son "hashable".
        """
        if self.hasheable(origen) and self.hasheable(destino):
            diccionario = self.dijkstra(origen)

            # Garantiza que el destino se encuentre en el grafo
            if diccionario[destino] is None:
                return []

            camino_correcto = []
            # Generamos el camino desde el destino hasta el origen mientras el destino no sea igual al origen
            while destino is not origen:
                camino_correcto.append(destino)
                destino = diccionario[destino]
            camino_correcto.append(destino)

            # Aplicamos un reverse ya que queremos el camino desde el origen al destino
            return list(reversed(camino_correcto))

    def prim(self):
        """Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.

        Args: None
        Returns:
            Dict[object,object]: Devuelve un diccionario que indica, para cada vértice del
                grafo, qué vértice es su padre en el árbol abarcador mínimo.
        Raises: None
        Example:
            Si G.prim()={2:1, 3:2, 4:1} entonces en un árbol abarcador mínimo tenemos que:
                1 es padre de 2 y de 4
                2 es padre de 3
        """
        # creamos el diccionario w
        w = {}
        for vertice in self.lista_vertices():
            for v2, (_, peso) in self.adyacencia[vertice].items():
                w[(vertice, v2)] = peso

        # incializamos las listas de padres y costes mínmos de aristas de cada vértice
        padre = {}
        costo_minimo = {}
        Q = []

        for v in self.lista_vertices():
            padre[v] = None
            costo_minimo[v] = INFTY
            heapq.heappush(Q, (costo_minimo[v], v))

        while Q:
            _, v = heapq.heappop(Q)

            # La interseccion la hago iterando por la lista de adyacientes a v y si tambien se encuentra en Q la añado a la lista
            lista_validos = []
            for vertice in self.adyacencia[v]:
                # Importante ponerlo como si fuera una tupla ya que en Q estan guardado en tuplas
                if (costo_minimo[vertice], vertice) in Q:
                    lista_validos.append(vertice)

            for x in lista_validos:
                if w[(v, x)] < costo_minimo[x]:
                    costo_minimo[x] = w[(v, x)]
                    padre[x] = v

                    # Actualizamos Q con el nuevo coste minimo
                    heapq.heappush(Q, (costo_minimo[x], x))

        # Como no nos interesa que salge el vertice None lo borramos
        for nodo in list(padre.keys()):
            if padre[nodo] == None:
                del padre[nodo]

        return padre

    def kruskal(self) -> List[Tuple[object, object]]:
        """Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Kruskal.

        Args: None
        Returns:
            List[Tuple[object,object]]: Devuelve una lista [(s1,t1),(s2,t2),...,(sn,tn)]
                de los pares de vértices del grafo que forman las aristas
                del arbol abarcador mínimo.
        Raises: None
        Example:
            En el ejemplo anterior en que G.kruskal()={2:1, 3:2, 4:1} podríamos tener, por ejemplo,
            G.prim=[(1,2),(1,4),(3,2)]
        """
        # Crear lista de aristas L ordenada por su peso c

        lista_aristas = []
        vertices_visitados = []
        for vertice in self.lista_vertices():
            vertices_adyacientes = self.adyacencia[vertice].keys()
            vertices_visitados.append(vertice)
            for vertice_adyaciente in vertices_adyacientes:
                if vertice_adyaciente not in vertices_visitados:
                    _, peso = self.adyacencia[vertice][vertice_adyaciente]
                    lista_aristas.append((peso, (vertice, vertice_adyaciente)))

        lista_aristas = sorted(lista_aristas)

        # lista: aristas aam
        aristas_aam = []

        # Diccionario
        diccionario = {}
        for vertice in self.lista_vertices():
            diccionario[vertice] = {vertice}

        # Recorremos la lista de aristas hasta que este vacia
        while len(lista_aristas) > 0:
            arista = lista_aristas[0][1]
            lista_aristas.remove(lista_aristas[0])

            if diccionario[arista[0]] != diccionario[arista[1]]:
                aristas_aam.append(arista)

                # Actualizamos el diccionario y hacemos la union
                diccionario[arista[0]] = diccionario[arista[0]].union(
                    diccionario[arista[1]]
                )

                for vertice in diccionario[arista[0]]:
                    diccionario[vertice] = diccionario[arista[0]]

        return aristas_aam

    #### NetworkX ####
    def convertir_a_NetworkX(self) -> nx.Graph or nx.DiGraph:
        """Construye un grafo o digrafo de Networkx según corresponda
        a partir de los datos del grafo actual.

        Args: None
        Returns:
            networkx.Graph: Objeto Graph de NetworkX si el grafo es no dirigido.
            networkx.DiGraph: Objeto DiGraph si el grafo es dirigido.
            En ambos casos, los vértices y las aristas son los contenidos en el grafo dado.
        Raises: None
        """
        # Primero vemos si es dirigido
        if self.es_dirigido():
            grafo = nx.DiGraph()
        else:
            grafo = nx.Graph()

        # Agregar vertices al grafo de NetworkX
        grafo.add_nodes_from(self.lista_vertices())

        aristas = []

        for vertice in self.lista_vertices():
            lista_adyacientes = self.lista_adyacencia(vertice)
            for vecino in lista_adyacientes:
                obj, peso = self.obtener_arista(vertice, vecino)
                informacion = {"object": obj, "weight": peso}
                aristas.append((vertice, vecino, informacion))

        # Agregar aristas al grafo de NetworkX
        grafo.add_edges_from(aristas)

        return grafo
