# Utility functions for Smart Agriculture Decision Support System

def get_zone_description(cluster_id):
    zones = {
        0: "Low Nutrient Zone",
        1: "High Moisture Zone",
        2: "Balanced Zone",
        3: "Dry Acidic Zone"
    }
    return zones.get(cluster_id, "Unknown Zone")

def get_feature_list():
    return ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]