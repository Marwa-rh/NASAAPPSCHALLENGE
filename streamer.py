import asyncio
import json
import os
import pandas as pd
import websockets

# 1. Dataset Ingestion & Cleaning
def load_and_clean_data(file_path="astronaut_vitals.csv"):
    if not os.path.exists(file_path):
        print(f"⚠️ Avertissement : {file_path} introuvable. Utilisation de données simulées par défaut.")
        # Données de secours si le CSV n'est pas encore créé
        return [
            {"heart_rate": 72, "spo2": 98, "suit_co2": 400, "radiation": 0.05, "core_temp": 37.0, "pressure": 101.3}
        ]
    
    df = pd.read_csv(file_path)
    
    # Dictionnaire de mapping pour standardiser les colonnes si le CSV original a d'autres noms
    # Modifie les clés de gauche selon les vrais noms de colonnes de ton CSV
    column_mapping = {
        # 'hr': 'heart_rate',
        # 'oxygen': 'spo2',
        # 'co2': 'suit_co2',
        # 'rad': 'radiation',
        # 'temp': 'core_temp',
        # 'press': 'pressure'
    }
    df = df.rename(columns=column_mapping)
    
    # Nettoyage basique (suppression des valeurs manquantes si besoin)
    df = df.ffill().fillna(0)
    
    return df.to_dict(orient="records")

# 3. Scenario Preset Triggers
SCENARIOS = {
    "NOMINAL": {},
    "HYPOXIA": {"suit_co2": 2450, "spo2": 87},
    "RADIATION": {"radiation": 0.85},
    "FATIGUE": {"heart_rate": 115, "core_temp": 38.1}
}

current_scenario = "NOMINAL"
connected_clients = set()

async def handle_client(websocket):
    global current_scenario
    connected_clients.add(websocket)
    print(f"🔌 Nouveau client connecté : {websocket.remote_address}")
    try:
        # Écoute les messages du client pour changer de scénario si nécessaire
        async for message in websocket:
            try:
                data = json.loads(message)
                if "command" in data and data["command"] in SCENARIOS:
                    current_scenario = data["command"]
                    print(f"🎯 Scénario appliqué : {current_scenario}")
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print("❌ Client déconnecté")

# 2. WebSocket Streamer (1Hz Loop)
async def broadcast_telemetry(dataset):
    global current_scenario
    index = 0
    while True:
        if not dataset:
            await asyncio.sleep(1)
            continue
            
        # Récupère la ligne courante du dataset (boucle infinie)
        row = dataset[index % len(dataset)].copy()
        index += 1

        # Applique les modifications du scénario actif
        if current_scenario in SCENARIOS:
            row.update(SCENARIOS[current_scenario])
        
        row["scenario"] = current_scenario

        # Diffuse à tous les clients connectés
        if connected_clients:
            payload = json.dumps(row)
            await asyncio.gather(
                *[client.send(payload) for client in connected_clients]
            )
            print(f"📤 Données envoyées [{current_scenario}] : {payload}")

        # Attente de 1000ms (1Hz)
        await asyncio.sleep(1.0)

async def main():
    dataset = load_and_clean_data()
    
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("🚀 Serveur WebSocket démarré sur ws://localhost:8765")
    
    # Lancement de la boucle de diffusion en arrière-plan
    broadcast_task = asyncio.create_task(broadcast_telemetry(dataset))
    
    await asyncio.gather(server.wait_closed(), broadcast_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du streamer.")