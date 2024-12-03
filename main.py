import os
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from pyrosm import OSM, get_data
import geopandas as gpd
import math

def calcular_distancia_euclidiana(nodo1, nodo2, G):
    """
    Calcula la distancia euclidiana entre dos nodos basándose en sus coordenadas.
    Suponemos que los nodos tienen atributos 'x' y 'y' con sus coordenadas en el plano.
    :param nodo1: Nodo de origen
    :param nodo2: Nodo de destino
    :param G: Grafo con los nodos y sus coordenadas
    :return: Distancia euclidiana entre los nodos
    """
    x1, y1 = G.nodes[nodo1]['x'], G.nodes[nodo1]['y']
    x2, y2 = G.nodes[nodo2]['x'], G.nodes[nodo2]['y']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def seleccionar_destino(nodo_inicial, nodos_evac, G):
    """
    Selecciona el nodo de evacuación más cercano considerando si hay camino.
    Si no hay camino hacia ningún nodo de evacuación, selecciona el nodo más cercano que sí tenga conexión.
    :param nodo_inicial: Nodo de origen
    :param nodos_evac: Lista de nodos de evacuación
    :param G: Grafo con las conexiones y coordenadas de los nodos
    :return: Nodo de evacuación más cercano
    """
    distancias = {}
    destinos_con_camino = []
    destino_seleccionado = None

    for nodo in nodos_evac:
        try:
            distancias[nodo] = nx.shortest_path_length(G, source=nodo_inicial, target=nodo, weight='weight')
            destinos_con_camino.append(nodo)  
        except nx.NetworkXNoPath:
            continue

    if destinos_con_camino:
        destino_seleccionado = min(destinos_con_camino, key=lambda nodo: distancias[nodo])
    else:
        distancias_a_evac = {}
        for nodo in nodos_evac:
            distancias_a_evac[nodo] = calcular_distancia_euclidiana(nodo_inicial, nodo, G)

        nodo_mas_cercano = min(distancias_a_evac, key=distancias_a_evac.get)

        for nodo in G.neighbors(nodo_inicial):
            try:
                nx.shortest_path_length(G, source=nodo, target=nodo_mas_cercano, weight='weight')
                destino_seleccionado = nodo_mas_cercano
                break
            except nx.NetworkXNoPath:
                continue

    return destino_seleccionado

def cargar_mapa(filepath, tipo='drive'):
    """
    Carga el mapa desde un archivo .osm o .osm.pbf local.
    :param filepath: Ruta al archivo .osm o .osm.pbf.
    :param tipo: Tipo de red (e.g., 'drive' para automóviles, 'walk' para peatones).
    :return: Grafo de la red (NetworkX)
    """
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.osm':
        # Cargar el grafo usando OSMnx
        G = ox.graph_from_xml(filepath, simplify=True, bidirectional=True)
    elif ext.lower() in ['.pbf', '.osm.pbf']:
        # Cargarlo utilizando pyrosm
        osm = OSM(filepath)
        
        # Seleccionar el tipo de red según el parámetro 'tipo'
        if tipo == 'drive':
            network_type = 'driving'
        elif tipo == 'walk':
            network_type = 'walking'
        else:
            network_type = 'all' 
        
        # nodes, edges = self.pyrosmOsm.get_network(nodes=True, network_type='driving')
        # self.graph = self.pyrosmOsm.to_graph(nodes, edges, graph_type='networkx', retain_all=True)

        if network_type != 'all':
            nodes, edges = osm.get_network(nodes=True, network_type=network_type)
        else:
            nodes, edges = osm.get_network(nodes=True, network_type='all')

        G = osm.to_graph(nodes, edges, graph_type='networkx', retain_all=True)
        
        if 'length' not in list(nx.get_edge_attributes(G, 'length').keys()):
            G = ox.distance.add_edge_lengths(G)
    else:
        raise ValueError("Formato de archivo no soportado. Usa .osm o .osm.pbf")
    
    if ext.lower() == '.osm' and tipo != 'all':
        # Estas son las calles que queremos mantener
        tipos_via = {
            'drive': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service'],
            'walk': ['footway', 'pedestrian', 'path', 'living_street', 'residential', 'service'],
        }

        tipos_filtrar = tipos_via.get(tipo, tipos_via['drive'])  # Por defecto 'drive'

        # Convertir el grafo a GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

        # Filtrar las aristas según el tipo de vía
        edges_filtradas = edges[edges['highway'].isin(tipos_filtrar)]

        G = ox.graph_from_gdfs(nodes, edges_filtradas, graph_attrs=G.graph)
    
    # Asegurarse de que todas las aristas tengan el atributo 'length'
    if 'length' not in list(nx.get_edge_attributes(G, 'length').keys()):
        G = ox.distance.add_edge_lengths(G)
    
    return G

def definir_puntos_evac(puntos, G):
    """
    Define los nodos de evacuación en el grafo.
    :param puntos: Lista de coordenadas (lat, lon).
    :param G: Grafo de la red
    :return: Lista de nodos más cercanos
    """
    nodos = []
    for punto in puntos:
        nodo = ox.distance.nearest_nodes(G, X=punto[1], Y=punto[0])
        nodos.append(nodo)
    return nodos

class Agente:
    def __init__(self, id, nodo_inicial, destino, velocidad=1.4):
        """
        Inicializa un agente.
        :param id: Identificador único.
        :param nodo_inicial: Nodo de inicio en el grafo.
        :param destino: Nodo de evacuación.
        :param velocidad: Velocidad en m/s.
        """
        self.id = id
        self.nodo_actual = nodo_inicial
        self.destino = destino
        self.velocidad = velocidad  # metros por segundo
        self.ruta = []
        self.tiempo_total = 0
        self.edge_progress = 0  # Distancia recorrida en la arista actual
        self.current_edge_length = 0  # Longitud de la arista actual

    def planificar_ruta(self, G):
        """
        Planifica la ruta más corta hacia el destino y almacena las longitudes de las aristas.
        :param G: Grafo de la red
        """
        try:
            self.ruta = nx.shortest_path(G, self.nodo_actual, self.destino, weight='length')
            self.aristas = []
            for u, v in zip(self.ruta[:-1], self.ruta[1:]):
                data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])
                self.aristas.append(data['length'])
        except nx.NetworkXNoPath:
            self.ruta = []
            self.aristas = []

    def mover(self, G, tiempo_intervalo):
        """
        Mueve el agente a lo largo de su ruta.
        :param G: Grafo de la red
        :param tiempo_intervalo: Tiempo de intervalo en segundos
        """
        if not self.ruta or self.nodo_actual == self.destino:
            return  

        distancia_a_mover = self.velocidad * tiempo_intervalo
        while distancia_a_mover > 0 and self.nodo_actual != self.destino:
            if not self.aristas:
                break 

            if self.current_edge_length == 0:
                self.current_edge_length = self.aristas[0]

            distancia_restante = self.current_edge_length - self.edge_progress

            print(distancia_a_mover)

            if distancia_a_mover >= distancia_restante:
                # Avanzar al siguiente nodo
                distancia_a_mover -= distancia_restante
                self.edge_progress = 0
                self.current_edge_length = 0
                self.nodo_actual = self.ruta[1]
                self.ruta.pop(0)
                self.aristas.pop(0)
                print(f"Agente {self.id} ha llegado a {self.nodo_actual}")
            else:
                # Avanzar parcialmente en la arista actual
                print(f"Agente {self.id} avanzó parcialmente {distancia_a_mover}")
                self.edge_progress += distancia_a_mover
                distancia_a_mover = 0


        self.tiempo_total += tiempo_intervalo

def simular_evac(G, agentes, tiempo_total, intervalo=1):
    """
    Simula la evacuación.
    :param G: Grafo de la red
    :param agentes: Lista de agentes
    :param tiempo_total: Tiempo total de simulación en segundos
    :param intervalo: Intervalo de tiempo en segundos
    :return: Registro de rutas de los agentes (coordenadas)
    """
    registros = defaultdict(list)

    for t in range(0, tiempo_total, intervalo):
        for agente in agentes:
            if agente.nodo_actual != agente.destino:
                agente.mover(G, intervalo)
                lat = G.nodes[agente.nodo_actual]['y']
                lon = G.nodes[agente.nodo_actual]['x']
                registros[agente.id].append((lat, lon))
        # Se pueden añadir condiciones de incendio que bloqueen rutas, modificando el grafo G en tiempo real
    return registros

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el mapa desde un archivo .osm
    # filepath = "CuraumaPlacilla.osm"
    # Cargar el mapa desde un archivo .osm.pbf (estos se sacan y se descargan de GeoFabrik, por lo general suelen estar mucho más limpios y precisos que sacarlos directamente de OSM)
    filepath = "CuraumaPlacilla.osm.pbf"  
    G = cargar_mapa(filepath, tipo='walk')  # En vez de 'walk' puede ser otro tipo, como 'drive'

    # 2. Definir puntos de evacuación (reemplaza con coordenadas reales)
    puntos_evac = [
        (-33.11113537001334, -71.56088105402401), # (lat, lng)
        (-33.12753092759561, -71.56761539484236),
    ]

    nodos_evac = definir_puntos_evac(puntos_evac, G)

    num_agentes = 100  # Número de personas a evacuar
    agentes = []
    for i in range(num_agentes):
        destino = None
        nodo_inicial = None
        while not destino:
            nodo_inicial = random.choice(list(G.nodes))
            
            # Llamar a la función seleccionar_destino para obtener el nodo de evacuación más cercano
            destino = seleccionar_destino(nodo_inicial, nodos_evac, G)
            
            # Verificar si se encontró un destino alcanzable
            if destino is None:
                print("No hay caminos alcanzables hacia los destinos de evacuación.")
            else:
                print(f"El destino de evacuación más cercano es: {destino}")

        # Crear y planificar la ruta del agente
        agente = Agente(id=i, nodo_inicial=nodo_inicial, destino=destino)
        agente.planificar_ruta(G)
        agentes.append(agente)

    tiempo_simulacion = 3600  # Simular una hora (3600 segundos)
    registros = simular_evac(G, agentes, tiempo_simulacion)

    # 5. Guardar los resultados con coordenadas
    # Crear un DataFrame donde cada agente tiene dos columnas: 'agent_{id}_lat' y 'agent_{id}_lon'
    data = {}
    num_filas = tiempo_simulacion // 1  # intervalo=1
    for agente_id, coordenadas in registros.items():
        lat_col = f"agent_{agente_id}_lat"
        lon_col = f"agent_{agente_id}_lon"
        # Rellenar con la última posición si la simulación terminó antes de tiempo_total
        latitudes = [coord[0] for coord in coordenadas]
        longitudes = [coord[1] for coord in coordenadas]
        if len(latitudes) < num_filas:
            latitudes.extend([latitudes[-1]] * (num_filas - len(latitudes)))
            longitudes.extend([longitudes[-1]] * (num_filas - len(longitudes)))
        data[lat_col] = latitudes
        data[lon_col] = longitudes

    df = pd.DataFrame(data)
    df.to_csv("rutas_evac_coordenadas.csv", index=False)
    print("Simulación completada. Resultados guardados en 'rutas_evac_coordenadas.csv'")
