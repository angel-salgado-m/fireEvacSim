import folium
import pandas as pd
import random
import os

def lat_lon_to_lon_lat(coord):
    """
    Convierte una tupla de coordenadas de (lat, lon) a (lon, lat).
    
    :param coord: Tupla de coordenadas (lat, lon)
    :return: Tupla de coordenadas (lon, lat)
    """
    lat, lon = coord
    return (lon, lat)

def visualizar_rutas(csv_filepath, mapa_salida="rutas_evac.html", num_agentes_mostrar=10):
    """
    Lee el archivo CSV con coordenadas y genera un mapa interactivo con las rutas de evacuación.
    
    :param csv_filepath: Ruta al archivo CSV con coordenadas.
    :param mapa_salida: Nombre del archivo HTML de salida.
    :param num_agentes_mostrar: Número de agentes a mostrar en el mapa.
    """
    # Verificar si el archivo existe
    if not os.path.exists(csv_filepath):
        print(f"El archivo '{csv_filepath}' no existe. Verifica la ruta y el nombre del archivo.")
        return

    # Cargar los datos
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    # Verificar que el CSV no esté vacío
    if df.empty:
        print("El archivo CSV está vacío. No hay datos para visualizar.")
        return

    # Obtener columnas de latitud y longitud
    lat_cols = [col for col in df.columns if col.endswith('_lat')]
    lon_cols = [col for col in df.columns if col.endswith('_lon')]

    if not lat_cols or not lon_cols:
        print("No se encontraron columnas de latitud y longitud en el CSV.")
        return

    # Calcular el centro del mapa usando el promedio de todas las coordenadas
    centro_lat = df[lat_cols].mean().mean()
    centro_lon = df[lon_cols].mean().mean()

    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=14)

    # Seleccionar aleatoriamente un subconjunto de agentes para visualizar
    agentes = [col.split('_')[1] for col in lat_cols]
    total_agentes = len(agentes)
    num_agentes_mostrar = min(num_agentes_mostrar, total_agentes)
    
    if num_agentes_mostrar < total_agentes:
        agentes_seleccionados = random.sample(agentes, num_agentes_mostrar)
    else:
        agentes_seleccionados = agentes

    print(f"Visualizando rutas de {len(agentes_seleccionados)} agentes out of {total_agentes}.")

    # Colores para las rutas (asegurarse de tener suficientes colores)
    colores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
               'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
               'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
               'gray', 'black', 'lightgray']

    for idx, agente_id in enumerate(agentes_seleccionados):
        lat_col = f"agent_{agente_id}_lat"
        lon_col = f"agent_{agente_id}_lon"

        # Verificar que ambas columnas existan
        if lat_col not in df.columns or lon_col not in df.columns:
            print(f"Las columnas {lat_col} o {lon_col} no existen en el CSV.")
            continue

        # Extraer la ruta del agente
        ruta_df = df[[lat_col, lon_col]].dropna()

        # Convertir a lista de listas (lat, lon)
        ruta_lat_lon = ruta_df.values.tolist()

        # Convertir a lista de listas (lon, lat) para OSMnx o NetworkX si es necesario
        # ruta_lon_lat = [lat_lon_to_lon_lat(coord) for coord in ruta_lat_lon]

        # Verificar que la ruta tenga al menos dos puntos
        if len(ruta_lat_lon) < 2:
            print(f"La ruta del agente {agente_id} no tiene suficientes puntos para trazar.")
            continue

        print(f"Trazando ruta para el agente {agente_id} con {len(ruta_lat_lon)} puntos.")

        # Seleccionar el color para la ruta
        color = colores[idx % len(colores)]

        # Trazar la ruta en el mapa usando (lat, lon)
        folium.PolyLine(
            ruta_lat_lon,
            color=color,
            weight=2.5,
            opacity=0.8,
            tooltip=f"Agente {agente_id}"
        ).add_to(mapa)

        # Añadir marcador al inicio de la ruta
        folium.Marker(
            location=ruta_lat_lon[0],
            popup=f"Inicio Agente {agente_id}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(mapa)

        # Añadir marcador al final de la ruta
        folium.Marker(
            location=ruta_lat_lon[-1],
            popup=f"Destino Agente {agente_id}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(mapa)

    # Guardar el mapa en un archivo HTML
    try:
        mapa.save(mapa_salida)
        print(f"Mapa interactivo guardado como '{mapa_salida}'.")
    except Exception as e:
        print(f"Error al guardar el mapa: {e}")

if __name__ == "__main__":
    # Ruta al archivo CSV generado por la simulación
    csv_filepath = "rutas_evac_coordenadas.csv"  # Reemplaza con la ruta real de tu archivo

    # Generar el mapa
    visualizar_rutas(csv_filepath, mapa_salida="rutas_evac.html", num_agentes_mostrar=20)
