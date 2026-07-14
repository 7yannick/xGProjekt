"""
Feature-Extraktion für das xG-Projekt.
Rechnet aus StatsBomb-Event-Rohdaten (location, freeze_frame) sinnvolle
ML-Features für ein Schuss (Shot).

StatsBomb-Koordinatensystem: Spielfeld 120 x 80 (Länge x Breite).
Das gegnerische Tor liegt bei x=120, y=40 (Mitte), Torpfosten bei y=36 und y=44.
"""
import math

GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_POST_1 = 36.0
GOAL_POST_2 = 44.0


def distance_to_goal(location):
    """Euklidische Distanz vom Schussort zur Torlinienmitte."""
    x, y = location[0], location[1]
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def angle_to_goal(location):
    """
    Sichtwinkel (in Grad) auf die Torbreite vom Schussort aus.
    Kernfeature jedes klassischen xG-Modells: je größer der Winkel,
    desto größer die 'sichtbare' Torfläche.
    """
    x, y = location[0], location[1]
    dx = GOAL_X - x
    dy1 = GOAL_POST_1 - y
    dy2 = GOAL_POST_2 - y
    if dx <= 0:
        return 0.0
    angle1 = math.atan2(dy1, dx)
    angle2 = math.atan2(dy2, dx)
    angle = abs(math.degrees(angle2 - angle1))
    return angle


def defenders_in_cone(location, freeze_frame):
    """
    Zählt gegnerische Spieler (inkl. Torwart), die sich näher am Schützen
    als am Tor befinden UND innerhalb des Sichtkegels zum Tor liegen.
    Grobe, aber gängige Approximation für 'wie zugestellt ist der Schuss'.
    """
    if not isinstance(freeze_frame, list):
        return None
    x, y = location[0], location[1]
    count = 0
    for p in freeze_frame:
        if p.get("teammate"):
            continue
        px, py = p["location"][0], p["location"][1]
        # nur Spieler zwischen Schütze und Tor berücksichtigen
        if px <= x:
            continue
        # Winkel-Check: liegt der Gegner ungefähr im Schusskegel Richtung Tor?
        shot_angle = math.atan2(GOAL_Y - y, GOAL_X - x)
        player_angle = math.atan2(py - y, px - x)
        if abs(math.degrees(shot_angle - player_angle)) < 8:
            count += 1
    return count


def goalkeeper_distance_to_goal(freeze_frame):
    """Distanz des gegnerischen Torwarts zur eigenen Torlinie."""
    if not isinstance(freeze_frame, list):
        return None
    for p in freeze_frame:
        if p.get("position", {}).get("name") == "Goalkeeper" and not p.get("teammate"):
            gx, gy = p["location"][0], p["location"][1]
            return math.hypot(GOAL_X - gx, GOAL_Y - gy)
    return None


def extract_shot_features(shot_row):
    """Nimmt eine Zeile aus dem StatsBomb-Events-DataFrame (type == 'Shot')
    und gibt ein dict mit allen ML-Features zurück."""
    loc = shot_row["location"]
    freeze = shot_row.get("shot_freeze_frame")

    return {
        "match_id": shot_row["match_id"],
        "player": shot_row.get("player"),
        "team": shot_row.get("team"),
        "minute": shot_row.get("minute"),
        "x": loc[0],
        "y": loc[1],
        "distance_to_goal": distance_to_goal(loc),
        "angle_to_goal": angle_to_goal(loc),
        "body_part": shot_row.get("shot_body_part"),
        "shot_type": shot_row.get("shot_type"),
        "technique": shot_row.get("shot_technique"),
        "under_pressure": bool(shot_row.get("under_pressure", False)),
        "n_defenders_in_cone": defenders_in_cone(loc, freeze),
        "gk_distance_to_goal": goalkeeper_distance_to_goal(freeze),
        "play_pattern": shot_row.get("play_pattern"),
        "statsbomb_xg": shot_row.get("shot_statsbomb_xg"),  # zum späteren Vergleich, NICHT als Feature nutzen!
        "is_goal": 1 if shot_row.get("shot_outcome") == "Goal" else 0,
    }
