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
from scipy.spatial import KDTree

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
        # Cargarlo utilizando pyrosm
        osm = OSM(filepath)
        
        # Seleccionar el tipo de red de calles a extraer
        if tipo == 'drive':
            network_type = 'driving'
        elif tipo == 'walk':
            network_type = 'walking'
        else:
            network_type = 'all' 
        
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
        # Calles a mantener
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
    def __init__(self, id, nodo_inicial, destino, tipo='normal', velocidad=1.4, tiempo_reaccion=0.0):
        """
        Inicializa un agente.
        :param id: Identificador único.
        :param nodo_inicial: Nodo de inicio en el grafo.
        :param destino: Nodo de evacuación.
        :param tipo: Tipo de agente ('normal' o 'abuelito').
        :param velocidad: Velocidad base en m/s.
        :param tiempo_reaccion: Tiempo de reacción en segundos.
        """
        self.id = id
        self.nodo_actual = nodo_inicial
        self.destino = destino
        self.tipo = tipo  # 'normal' o 'abuelito'
        self.base_velocidad = velocidad if tipo == 'normal' else 1.0  # Abuelitos tienen velocidad menor
        self.velocidad_actual = self.base_velocidad  # Velocidad ajustada según densidad
        self.ruta = []
        self.tiempo_total = 0
        self.edge_progress = 0  # Distancia recorrida en la arista actual
        self.current_edge_length = 0  # Longitud de la arista actual


        self.distancia_recorrida = 0.0  # Total de metros recorridos
        self.evacuado = False  # Indicador de si ha evacuado
        self.tiempo_evacuado = None  # Tiempo de evacuación
        self.zona_evac_reached = None  # Zona de evacuación alcanzada
        self.velocidad_acumulada = 0.0  # Velocidades promedio
        self.intervalos = 0  
        self.tiempo_reaccion = tiempo_reaccion  # Tiempo de reacción (s)
        self.inicio_movimiento = False

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

    def mover(self, G, tiempo_intervalo, tiempo_actual):
        """
        Mueve el agente a lo largo de su ruta y actualiza las métricas.
        :param G: Grafo de la red
        :param tiempo_intervalo: Tiempo de intervalo en segundos
        :param tiempo_actual: Tiempo actual de la simulación en segundos
        """
        if self.evacuado:
            return

        # Verificar si el agente ha superado su tiempo de reacción
        if not self.inicio_movimiento:
            if tiempo_actual >= self.tiempo_reaccion:
                self.inicio_movimiento = True
            else:
                # Incrementar el tiempo total sin mover
                self.tiempo_total += tiempo_intervalo
                return

        if not self.ruta or self.nodo_actual == self.destino:
            return  

        distancia_a_mover = self.velocidad_actual * tiempo_intervalo
        while distancia_a_mover > 0 and self.nodo_actual != self.destino:
            if not self.aristas:
                break 

            if self.current_edge_length == 0:
                self.current_edge_length = self.aristas[0]

            distancia_restante = self.current_edge_length - self.edge_progress

            if distancia_a_mover >= distancia_restante:
                # Avanzar al siguiente nodo
                self.distancia_recorrida += distancia_restante
                distancia_a_mover -= distancia_restante
                self.edge_progress = 0
                self.current_edge_length = 0
                self.nodo_actual = self.ruta[1]
                self.ruta.pop(0)
                self.aristas.pop(0)
            else:
                # Avanzar parcialmente en la arista actual
                self.distancia_recorrida += distancia_a_mover
                self.edge_progress += distancia_a_mover
                distancia_a_mover = 0

        self.tiempo_total += tiempo_intervalo
        self.velocidad_acumulada += self.velocidad_actual
        self.intervalos += 1

        if self.nodo_actual == self.destino and not self.evacuado:
            self.evacuado = True
            self.tiempo_evacuado = tiempo_actual
            self.zona_evac_reached = self.destino

    def calcular_velocidad_promedio(self):
        """
        Calcula la velocidad promedio del agente.
        :return: Velocidad promedio en m/s
        """
        if self.tiempo_total > 0:
            return self.distancia_recorrida / self.tiempo_total
        else:
            return 0.0

def simular_evac(G, agentes, tiempo_total, intervalo=1, radio=50):
    """
    Simula la evacuación.
    :param G: Grafo de la red
    :param agentes: Lista de agentes
    :param tiempo_total: Tiempo total de simulación en segundos
    :param intervalo: Intervalo de tiempo en segundos
    :param radio: Radio en metros para considerar la densidad de agentes
    :return: Registro de rutas de los agentes (coordenadas) y métricas
    """
    registros = defaultdict(list)
    metrics = {}

    for t in range(0, tiempo_total, intervalo):
        posiciones = []
        agentes_activos = []
        for agente in agentes:
            if not agente.evacuado:
                lat = G.nodes[agente.nodo_actual]['y']
                lon = G.nodes[agente.nodo_actual]['x']
                posiciones.append((lat, lon))
                agentes_activos.append(agente)
            else:
                lat = G.nodes[agente.nodo_actual]['y']
                lon = G.nodes[agente.nodo_actual]['x']
                posiciones.append((lat, lon))  # Mantener la última posición

        if agentes_activos:
            # Convertir lat/lon a coordenadas proyectadas para calcular distancias en metros
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[1] for p in posiciones],
                                                               [p[0] for p in posiciones]),
                                   crs='EPSG:4326')
            gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator para cálculos en metros
            coords = np.array([(point.x, point.y) for point in gdf_projected.geometry])

            tree = KDTree(coords)

            # Para cada agente activo, encontrar la cantidad de agentes cercanos
            for idx, agente in enumerate(agentes_activos):
                indices = tree.query_ball_point(coords[idx], r=radio) # Agentes cercanos
                num_cercanos = len(indices) - 1 # se resta el mismo agente
                # Umbral de densidad para reducir velocidad
                if num_cercanos > 5:
                    factor = max(0.5, 1.0 - 0.05 * num_cercanos)  # Limitar la reducción al 50%
                    agente.velocidad_actual = agente.base_velocidad * factor
                else:
                    agente.velocidad_actual = agente.base_velocidad

        # Mover a los agentes
        for agente in agentes:
            if not agente.evacuado:
                agente.mover(G, intervalo, t)
                lat = G.nodes[agente.nodo_actual]['y']
                lon = G.nodes[agente.nodo_actual]['x']
                registros[agente.id].append((lat, lon))
            else:
                # Mantener la última posición después de evacuar
                lat = G.nodes[agente.nodo_actual]['y']
                lon = G.nodes[agente.nodo_actual]['x']
                registros[agente.id].append((lat, lon))

    for agente in agentes:
        velocidad_promedio = agente.calcular_velocidad_promedio()
        metrics[agente.id] = {
            'id': agente.id,
            'tipo': agente.tipo,
            'nodo_inicial': agente.ruta[0] if agente.ruta else None,
            'destino': agente.destino,
            'distancia_recorrida_m': agente.distancia_recorrida,
            'tiempo_total_s': agente.tiempo_total,
            'tiempo_evacuado_s': agente.tiempo_evacuado,
            'zona_evac_reached': agente.zona_evac_reached,
            'evacuado': agente.evacuado,
            'velocidad_promedio_m_s': velocidad_promedio,
            'tiempo_reaccion_s': agente.tiempo_reaccion
        }

    return registros, metrics

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el mapa desde un archivo .osm.pbf (estos se sacan y se descargan de GeoFabrik, por lo general suelen estar mucho más limpios y precisos que sacarlos directamente de OSM)
    filepath = "CuraumaPlacilla.osm.pbf"  
    G = cargar_mapa(filepath, tipo='walk')  # En vez de 'walk' puede ser otro tipo, como 'drive'

    # Puntos de evacuación
    puntos_evac = [
        (-33.11113537001334, -71.56088105402401), # (lat, lng)
        (-33.12753092759561, -71.56761539484236),
    ]

    nodos_evac = definir_puntos_evac(puntos_evac, G)

    num_agentes = 100  # Número de personas a evacuar
    prob_abuelito = 0.1  # 10% de probabilidad de ser abuelito
    agentes = []
    for i in range(num_agentes):
        destino = None
        nodo_inicial = None
        intentos = 0
        max_intentos = 1000  # Para evitar bucles infinitos
        while not destino and intentos < max_intentos:
            nodo_inicial = random.choice(list(G.nodes))
            
            # Llamar a la función seleccionar_destino para obtener el nodo de evacuación más cercano
            destino = seleccionar_destino(nodo_inicial, nodos_evac, G)
            
            # Verificar si se encontró un destino alcanzable
            if destino is None:
                print("No hay caminos alcanzables hacia los destinos de evacuación.")
            else:
                print(f"El destino de evacuación más cercano es: {destino}")
            intentos += 1

        if intentos == max_intentos:
            print(f"Agente {i} no pudo encontrar un destino alcanzable. Se omitirá.")
            continue

        tipo = 'abuelito' if random.random() < prob_abuelito else 'normal'

        velocidad = 1.0 if tipo == 'abuelito' else 1.4  # m/s

        # Asignar tiempo de reacción basado en el tipo
        if tipo == 'abuelito':
            tiempo_reaccion = random.uniform(10, 30) # 10 a 30 segundos
        else:
            tiempo_reaccion = random.uniform(0, 10) # 0 a 10 segundos

        # Crear y planificar la ruta del agente
        agente = Agente(id=i, nodo_inicial=nodo_inicial, destino=destino, tipo=tipo, velocidad=velocidad, tiempo_reaccion=tiempo_reaccion)
        agente.planificar_ruta(G)
        agentes.append(agente)

    tiempo_simulacion = 3600  # Simular una hora (3600 segundos)
    radio_influencia = 50  # Radio en metros para considerar la densidad de agentes
    registros, metrics = simular_evac(G, agentes, tiempo_simulacion, radio=radio_influencia)

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

    df_rutas = pd.DataFrame(data)
    df_rutas.to_csv("rutas_evac_coordenadas.csv", index=False)
    print("Simulación completada. Resultados de rutas guardados en 'rutas_evac_coordenadas.csv'")

    # Métricas agentes
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.to_csv("metricas_agentes_evac.csv", index=False)
    print("Simulación completada. Métricas de agentes guardadas en 'metricas_agentes_evac.csv'")

    df_metrics = pd.read_csv("metricas_agentes_evac.csv")

    num_evacuados = df_metrics['evacuado'].sum()
    num_no_evacuados = len(df_metrics) - num_evacuados
    tiempo_promedio_evac = df_metrics[df_metrics['evacuado']]['tiempo_evacuado_s'].mean()
    distancia_promedio = df_metrics['distancia_recorrida_m'].mean()
    velocidad_promedio = df_metrics['velocidad_promedio_m_s'].mean()
    tiempo_reaccion_promedio = df_metrics['tiempo_reaccion_s'].mean()

    print(f"Número de agentes evacuados: {num_evacuados}")
    print(f"Número de agentes no evacuados: {num_no_evacuados}")
    print(f"Tiempo promedio de evacuación: {tiempo_promedio_evac:.2f} segundos")
    print(f"Distancia promedio recorrida: {distancia_promedio:.2f} metros")
    print(f"Velocidad promedio de todos los agentes: {velocidad_promedio:.2f} m/s")
    print(f"Tiempo de reacción promedio: {tiempo_reaccion_promedio:.2f} segundos")

    # Métricas específicas para los abuelitos
    df_abuelitos = df_metrics[df_metrics['tipo'] == 'abuelito']
    num_abuelitos_evacuados = df_abuelitos['evacuado'].sum()
    tiempo_promedio_abuelitos = df_abuelitos[df_abuelitos['evacuado']]['tiempo_evacuado_s'].mean()
    distancia_promedio_abuelitos = df_abuelitos['distancia_recorrida_m'].mean()
    velocidad_promedio_abuelitos = df_abuelitos['velocidad_promedio_m_s'].mean()
    tiempo_reaccion_promedio_abuelitos = df_abuelitos['tiempo_reaccion_s'].mean()

    print(f"Número de abuelitos evacuados: {num_abuelitos_evacuados}")
    print(f"Tiempo promedio de evacuación de abuelitos: {tiempo_promedio_abuelitos:.2f} segundos")
    print(f"Distancia promedio recorrida por abuelitos: {distancia_promedio_abuelitos:.2f} metros")
    print(f"Velocidad promedio de abuelitos: {velocidad_promedio_abuelitos:.2f} m/s")
    print(f"Tiempo de reacción promedio de abuelitos: {tiempo_reaccion_promedio_abuelitos:.2f} segundos")
