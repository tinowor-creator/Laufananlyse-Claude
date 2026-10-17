#!/usr/bin/env python3
"""Lauf-Coach (lokal): analysiert deinen Laufstil aus Seitenvideos.

Installation:  pip install mediapipe opencv-python numpy
Benutzung:     python lauf_coach.py lauf1.mp4 lauf2.mp4 --groesse 195 --gewicht 77 --schritt 97
Optional:      --skelett   schreibt zusätzlich Videos mit eingezeichnetem Skelett

Filmtipps: von der Seite, Handy fest, ganzer Körper im Bild, 10 bis 20 Sekunden.
Alle Werte sind Schätzungen aus einem 2D-Video, keine medizinische Diagnose.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

VERLAUF = Path("lauf_verlauf.json")
MODELL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
SCHLUESSEL = [0, 11, 12, 23, 24, 25, 26, 27, 28]
VERBINDUNGEN = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28), (27, 31), (28, 32)]


def metriken(h):
    c = 175 - .35 * (h - 175)  # größere Menschen laufen meist etwas langsamer getaktet
    return [
        dict(k="cad", n="Schrittfrequenz", u="Schritte/min", g=(c - 10, c + 10), o=(c - 20, c + 20),
             lo="Eher niedrig: kürzere, schnellere Schritte probieren (ca. 5 % mehr Schritte pro Minute), z. B. mit Metronom.",
             hi="Sehr hoch: wenn es anstrengend wirkt, etwas längere, entspanntere Schritte probieren."),
        dict(k="lean", n="Oberkörper-Neigung", u="°", g=(3, 10), o=(0, 14),
             lo="Sehr aufrecht oder nach hinten geneigt: locker aus den Knöcheln nach vorn neigen, Blick 10 bis 20 m voraus.",
             hi="Starke Vorneigung: aufrichten, Rumpf leicht anspannen, Kopf und Schultern locker über der Hüfte."),
        dict(k="over", n="Fuß landet vor dem Körper", u="% der Beinlänge", g=(0, 30), o=(0, 42),
             lo="", hi="Der Fuß greift weit nach vorn und bremst. Schrittfrequenz erhöhen, Fuß näher unter dem Körper aufsetzen."),
        dict(k="vert", n="Auf und ab pro Schritt", u="% der Rumpflänge", g=(0, 16), o=(0, 22),
             lo="", hi="Relativ viel Auf und Ab: „leise“ laufen, kleinere und schnellere Schritte, weniger Abdruck nach oben."),
        dict(k="kc", n="Knie beim Aufsetzen", u="° Beugung", g=(8, 25), o=(3, 32),
             lo="Das Knie ist beim Aufsetzen fast durchgestreckt und stößt hart. Fuß näher unter dem Körper aufsetzen, Knie leicht gebeugt lassen.",
             hi="Das Knie ist beim Aufsetzen stark gebeugt. Oft hilft mehr Spannung in Rumpf, Hüfte und Oberschenkel."),
        dict(k="sh", n="Schienbein beim Aufsetzen", u="° zur Senkrechten", g=(0, 10), o=(-3, 16),
             lo="", hi="Das Schienbein zeigt beim Aufsetzen weit nach vorn, der Fuß landet vor dem Knie. Schrittfrequenz erhöhen, Fuß näher unter die Hüfte bringen."),
        dict(k="kf", n="Fersenhub (max. Kniebeugung)", u="°", g=(80, 115), o=(65, 130),
             lo="Die Ferse kommt wenig Richtung Gesäß. Bei lockerem Tempo normal, bei schnellerem hilft es, das Knie aktiv nach vorn zu bringen.",
             hi="Sehr hoher Fersenhub. Falls du nicht schnell läufst, Beine entspannter und flacher durchschwingen."),
        dict(k="el", n="Ellbogenwinkel", u="°", g=(75, 105), o=(60, 120),
             lo="Die Arme sind stark angewinkelt. Locker mit entspannten Händen auf Hüfthöhe schwingen.",
             hi="Die Arme sind sehr gestreckt. Ellbogen etwa im rechten Winkel halten und vor und zurück schwingen, nicht quer."),
    ]


def abstand(m, v):
    return m["g"][0] - v if v < m["g"][0] else v - m["g"][1] if v > m["g"][1] else 0


def status(m, v):
    if abstand(m, v) == 0:
        return 0
    return 1 if m["o"][0] <= v <= m["o"][1] else 2


def zielbereich(m):
    g0, g1 = round(m["g"][0]), round(m["g"][1])
    return f"bis {g1}" if g0 <= 0 else f"{g0} bis {g1}"


def winkel(a, b, c):
    """Winkel bei b in Grad; a, b, c sind Arrays (N, 2)."""
    u, v = a - b, c - b
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def glatt(a, w):
    return np.convolve(np.pad(a, w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), mode="valid")


def analyse(t, P, hoehe):
    """t: Zeiten (s), P: Array (N, 33, 3) mit x, y in Pixeln und Sichtbarkeit."""
    t, P = np.asarray(t), np.asarray(P)
    gut = np.all(P[:, SCHLUESSEL, 2] > .5, axis=1)
    t, P = t[gut], P[gut]
    if len(t) < 20:
        return None
    xy = P[..., :2]
    sh, hp = (xy[:, 11] + xy[:, 12]) / 2, (xy[:, 23] + xy[:, 24]) / 2
    richtung = np.sign(np.median(xy[:, 0, 0] - hp[:, 0])) or 1.0
    rumpf = np.median(np.linalg.norm(sh - hp, axis=1))
    d = lambda a, b: np.linalg.norm(xy[:, a] - xy[:, b], axis=1)
    bein = np.median((d(23, 25) + d(25, 27) + d(24, 26) + d(26, 28)) / 2)
    lean = np.median(np.degrees(np.arctan2(richtung * (sh[:, 0] - hp[:, 0]), hp[:, 1] - sh[:, 1])))
    y = hp[:, 1]
    dt = y - glatt(y, 10)
    hub = np.percentile(dt, 95) - np.percentile(dt, 5)
    # Schrittfrequenz: Beine kreuzen sich zweimal pro Doppelschritt
    sm = glatt(xy[:, 27, 0] - xy[:, 28, 0], 1)
    thr, s, zeiten = .05 * bein, 0, []
    for i, v in enumerate(sm):
        if v > thr:
            if s < 0:
                zeiten.append(t[i])
            s = 1
        elif v < -thr:
            if s > 0:
                zeiten.append(t[i])
            s = -1
    cad = (len(zeiten) - 1) / (zeiten[-1] - zeiten[0]) * 60 if len(zeiten) >= 4 else None
    # Fuß vor dem Körper: größte Vorwärtsdistanz Knöchel zu Hüfte je Schritt
    gipfel, kc, sh_, kf = [], [], [], []
    for a, h in ((27, 23), (28, 24)):
        q = richtung * (xy[:, a, 0] - xy[:, h, 0]) / bein
        idx = [i for i in range(1, len(q) - 1) if q[i] > q[i - 1] and q[i] >= q[i + 1] and q[i] > .15]
        gipfel += [q[i] for i in idx]
        beugung = 180 - winkel(xy[:, h], xy[:, a - 2], xy[:, a])  # Kniebeugung, 0 = gestreckt
        kc += [beugung[i] for i in idx]
        sh_ += [np.degrees(np.arctan2(richtung * (xy[i, a, 0] - xy[i, a - 2, 0]), xy[i, a, 1] - xy[i, a - 2, 1])) for i in idx]
        kf += [beugung[i] for i in range(1, len(beugung) - 1)
               if beugung[i] > beugung[i - 1] and beugung[i] >= beugung[i + 1] and beugung[i] > 60]
    over = np.mean(gipfel) * 100 if len(gipfel) >= 3 else None
    arme = []
    for s_, e_, w_ in ((11, 13, 15), (12, 14, 16)):
        ok_ = P[:, [s_, e_, w_], 2].min(axis=1) > .5
        if ok_.sum() >= 20:
            arme.append((int(ok_.sum()), float(np.median(winkel(xy[ok_, s_], xy[ok_, e_], xy[ok_, w_])))))
    med = lambda l: np.median(l) if len(l) >= 3 else None
    massstab = (.288 * hoehe / rumpf + .491 * hoehe / bein) / 2  # cm pro Pixel
    r = lambda v: None if v is None else round(float(v), 1)
    return dict(cad=r(cad), lean=r(lean), over=r(over), vert=r(hub / rumpf * 100), vcm=r(hub * massstab),
                kc=r(med(kc)), sh=r(med(sh_)), kf=r(med(kf)), el=r(max(arme)[1] if arme else None))


def pose_funktion():
    """Liefert f(rgb, zeit_s) -> Liste (x, y, sichtbar) normiert oder None."""
    import mediapipe as mp
    if hasattr(mp, "solutions"):  # ältere MediaPipe-Versionen, Modell im Paket enthalten
        p = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5)

        def f(rgb, ts):
            res = p.process(rgb)
            return [(l.x, l.y, l.visibility) for l in res.pose_landmarks.landmark] if res.pose_landmarks else None
        return f
    from mediapipe.tasks import python as mpp  # neuere Versionen: Modelldatei wird einmalig geladen
    from mediapipe.tasks.python import vision
    modell = Path("pose_landmarker_full.task")
    if not modell.exists():
        import urllib.request
        print("Lade Pose-Modell (einmalig) ...")
        urllib.request.urlretrieve(MODELL_URL, modell)
    opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(modell)),
                                        running_mode=vision.RunningMode.VIDEO)
    lm = vision.PoseLandmarker.create_from_options(opts)

    def f(rgb, ts):
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts * 1000))
        return [(l.x, l.y, l.visibility) for l in res.pose_landmarks[0]] if res.pose_landmarks else None
    return f


def posen(pfad, pose, skelett):
    import cv2
    cap = cv2.VideoCapture(str(pfad))
    if not cap.isOpened():
        sys.exit(f"Video nicht lesbar: {pfad}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    schritt, maxn = max(1, round(fps / 20)), int(20 * fps)  # ca. 20 Bilder/s, erste 20 s
    t, P, writer, i = [], [], None, 0
    while i < maxn:
        ok, bild = cap.read()
        if not ok:
            break
        if i % schritt == 0:
            h, w = bild.shape[:2]
            lm = pose(cv2.cvtColor(bild, cv2.COLOR_BGR2RGB), i / fps)
            if lm:
                a = np.array([[x * w, y * h, v] for x, y, v in lm])
                t.append(i / fps)
                P.append(a)
                for p, q in VERBINDUNGEN:
                    cv2.line(bild, tuple(a[p, :2].astype(int)), tuple(a[q, :2].astype(int)), (232, 196, 92), 3)
            if skelett:
                if writer is None:
                    ziel = pfad.with_name(pfad.stem + "_skelett.mp4")
                    writer = cv2.VideoWriter(str(ziel), cv2.VideoWriter_fourcc(*"mp4v"), fps / schritt, (w, h))
                writer.write(bild)
        i += 1
    cap.release()
    if writer:
        writer.release()
    return t, P


def tempo(a, cad):
    kmh = a.schritt * cad * .0006
    s = round(3600 / kmh)
    return kmh, f"{s // 60}:{s % 60:02d}"


def bericht(name, r, M, a):
    print(f"\n=== {name} ===")
    tipps = []
    for m in M:
        v = r[m["k"]]
        if v is None:
            print(f"  {m['n']}: nicht messbar")
            continue
        s = status(m, v)
        print(f"  {m['n']}: {v} {m['u']}  [{['im Zielbereich', 'okay', 'auffällig'][s]}, Ziel {zielbereich(m)}]")
        if s and (m["lo"] if v < m["g"][0] else m["hi"]):
            tipps.append(m["lo"] if v < m["g"][0] else m["hi"])
    print("\n  Verbesserungsvorschläge:")
    for x in tipps or ["Alle Werte liegen im Zielbereich. Prüf in ein paar Wochen, ob es so bleibt."]:
        print(f"   - {x}")
    print("\n  Auf deinen Körper umgerechnet:")
    if r["cad"]:
        kmh, pace = tempo(a, r["cad"])
        print(f"   - Mit {a.schritt:.0f} cm Schrittlänge und {r['cad']} Schritten/min: ca. {kmh:.1f} km/h ({pace} min/km)")
    print(f"   - Schrittlänge = {a.schritt / a.groesse * 100:.1f} % der Körpergröße, ca. {round(100000 / a.schritt)} Schritte pro km")
    if r["over"] is not None:
        print(f"   - Fuß landet ca. {round(r['over'] / 100 * .491 * a.groesse)} cm vor der Hüfte")
    print(f"   - Hüfte bewegt sich pro Schritt ca. {r['vcm']} cm auf und ab")
    print(f"   - Landekraft grob das 2- bis 3-Fache des Körpergewichts: {round(a.gewicht * 2 * 9.81)} bis {round(a.gewicht * 3 * 9.81)} N")


def vergleich(titel, spalten, M):
    print(f"\n=== {titel} ===")
    print(f"{'Wert':28}" + "".join(f"{n[:18]:>20}" for n in spalten))
    for m in M:
        zeile = ""
        for r in spalten.values():
            v = r.get(m["k"])
            zeile += f"{'-' if v is None else str(v) + ' ' + '✓~!'[status(m, v)]:>20}"
        print(f"{m['n']:28}{zeile}")
    print("✓ im Zielbereich, ~ okay, ! auffällig")


def main():
    ap = argparse.ArgumentParser(description="Lauf-Coach: Laufstil aus Seitenvideos analysieren")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--groesse", type=float, default=195, help="Körpergröße in cm")
    ap.add_argument("--gewicht", type=float, default=77, help="Gewicht in kg")
    ap.add_argument("--schritt", type=float, default=97, help="Schrittlänge in cm")
    ap.add_argument("--skelett", action="store_true", help="Videos mit Skelett speichern")
    a = ap.parse_args()
    M = metriken(a.groesse)
    try:
        pose = pose_funktion()
    except ImportError:
        sys.exit("Bitte installieren: pip install mediapipe opencv-python numpy")
    neu = {}
    for v in a.videos:
        pfad = Path(v)
        print(f"\nAnalysiere {pfad.name} ...")
        t, P = posen(pfad, pose, a.skelett)
        r = analyse(t, P, a.groesse)
        if r is None:
            print("  Zu wenige Körperpunkte erkannt. Bitte seitlich und mit ganzem Körper im Bild filmen.")
            continue
        neu[pfad.name] = r
        bericht(pfad.name, r, M, a)
    if len(neu) > 1:
        vergleich("Vergleich dieser Videos", neu, M)
    if neu:
        try:
            alt = json.loads(VERLAUF.read_text()) if VERLAUF.exists() else []
        except ValueError:
            alt = []
        if alt:
            letzte = alt[-1]
            for name, r in neu.items():
                vergleich(f"{name} im Vergleich zu {letzte['name']} ({letzte['datum']})", {"jetzt": r, "früher": letzte["r"]}, M)
        jetzt = datetime.now().strftime("%d.%m.%Y %H:%M")
        alt += [dict(datum=jetzt, name=n, r=r) for n, r in neu.items()]
        VERLAUF.write_text(json.dumps(alt, ensure_ascii=False, indent=1))
        print(f"\nErgebnisse gespeichert in {VERLAUF}")


if __name__ == "__main__":
    main()
