# Integration: Stage 1 V2 + FIN (Hubraum)

> **Autor:** Moritz Binder · **Stand:** 28.06.2026 · **Status:** Modell trainiert, im Branch `Mail_project_moritz`

Dieses Dokument hält fest, wie die zwei Entwicklungsstränge des Teams zusammengeführt wurden und was dabei herauskam.

---

## TL;DR

Wir haben **Pascals V2-Modell** (reiche Manheim-Merkmale: trim, transmission, state, color, interior, make_model) mit **Moritz' FIN-Anreicherung** (Hubraum aus der VIN) kombiniert. Ergebnis auf dem vollen Datensatz (529.169 Zeilen):

| Modell | MAE | R² | MAPE |
|---|---:|---:|---:|
| Altes schlankes Stage 1 (HistGB) | $1.824 | 0,885 | 16,3 % |
| V2 (Pascal, reiche Merkmale) | $1.365 | 0,937 | 15,0 % |
| **V2 + Hubraum (integriert)** | **$1.201** | **0,954** | 13,7 % |

→ Das integrierte Modell ist **der beste Stage-1-Wert des Projekts**. Der Hubraum bringt **+12 %** zusätzlich zu V2 — also eigenständiges Signal, das sich nicht mit trim/Farbe etc. überschneidet.

---

## Die zwei Stränge

### Strang A — V2 (Pascal)
- Branch-Ursprung: `main` / `mail_project_pascal`
- Skript: `scripts/train_stage1_alt_production.py`
- Idee: das schlanke Stage-1-Modell um die bislang weggelassenen Manheim-Merkmale erweitern (Ausstattung, Getriebe, Bundesstaat, Außen-/Innenfarbe, Marke-Modell-Interaktion).
- Modell: XGBoost VotingRegressor 50/50 (`reg:squarederror` roh + `reg:absoluteerror` log).
- Ergebnis: MAE $1.370 (−25 % gegenüber dem schlanken Modell).

### Strang B — FIN/Hubraum (Moritz)
- Branch: `Mail_project_moritz`, Ordner `vin_fin_enrichment/`
- Idee: aus der VIN (US-FIN) über die NHTSA-API technische Merkmale nachladen. Ablationen zeigten: **der Hubraum trägt den gesamten Effekt** (Kraftstoff/Zylinder redundant).
- Datenbasis: alle 550.245 VINs dekodiert → `vin_decoded_cache_full.csv` (Hubraum-Abdeckung 99 %).

---

## Die Zusammenführung

Neues Skript: **`scripts/train_stage1_v2_fin.py`**

- Identisch zu `train_stage1_alt_production.py` (gleiche Features, Hyperparameter, Filter, Split, Ensemble).
- **Einzige Änderung:** `displacement` (Hubraum) wird als numerisches Feature ergänzt, per VIN aus `vin_decoded_cache_full.csv` gejoint.
- Fehlende Hubraum-Werte (~1 %) werden mit dem Median gefüllt.
- Ausgabe: `models/price_model_v2_fin.joblib` + `models/price_model_v2_fin_metrics.json`.

Reproduzieren:
```bash
uv run python scripts/train_stage1_v2_fin.py --max-rows 0   # voller Datensatz
```

---

## Belege (alle Tests im Überblick)

Der FIN/Hubraum-Effekt wurde über mehrere Stichprobengrößen und Modelle bestätigt:

| Test | Vergleich | Stichprobe | Effekt Hubraum |
|---|---|---|---|
| 3 | vs. einfache Baseline (HistGB) | 12k | +10,6 % |
| 4 | vs. V2-Features (HistGB) | 14k | +20,8 % |
| 5 | vs. V2-Features (HistGB) | 96k | +19,7 % |
| 6 | Per-Feature (welches Merkmal trägt?) | 96k | Hubraum allein = alle drei |
| 7 | vs. schlankes Stage-1 (HistGB) | **534k voll** | +13,1 % |
| 8 | vs. echtes V2-Ensemble (XGBoost) | **529k voll** | **+12,2 %** |

→ Konsistentes, robustes Ergebnis. Der Effekt schrumpft mit mehr Daten leicht (von ~20 % auf ~12 %), bleibt aber auch gegen das voll ausgereizte V2 klar bestehen.

---

## Status & nächste Schritte

**Erledigt:**
- [x] Alle 550k VINs dekodiert, Cache gesichert
- [x] FIN-Effekt über 6 Tests belegt (Hubraum als Treiber identifiziert)
- [x] Integriertes Modell V2 + Hubraum trainiert (`price_model_v2_fin.joblib`, MAE $1.201)
- [x] Dokumentiert (diese Datei + `FIN_Test.md`)

**Offen:**
- [ ] **Git-Branch-Merge** `main` ↔ `Mail_project_moritz` (main ist 23 Commits voraus: Repo-Reorg + V2; Moritz 14 voraus: FIN). Vorher: Moritz' Quantil-WIP committen. Konflikte erwartbar in `Aktueller_Stand.md`, `.gitignore`.
- [ ] Integriertes Modell in Streamlit-App + Stage 2/3 einbinden (braucht zusätzliche Eingabefelder bzw. VIN-Lookup).
- [ ] Im LNCS-Paper als belegte Feature-Erweiterung dokumentieren.

---

## Dateien dieser Integration

| Datei | Inhalt |
|---|---|
| `scripts/train_stage1_v2_fin.py` | Integrierte Trainings-Pipeline (V2 + Hubraum) |
| `models/price_model_v2_fin.joblib` | Trainiertes integriertes Modell (13 MB) |
| `models/price_model_v2_fin_metrics.json` | Metriken + Segmentfehler |
| `vin_fin_enrichment/` | Komplette FIN-Voruntersuchung (Tests 1–8, Cache, Doku) |
