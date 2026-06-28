# Experiment: VIN/FIN-Anreicherung über die NHTSA-API

> **Autor:** Moritz Binder · **Stand:** 24.06.2026 · **Status:** explorativ, nicht in Produktion
> **Branch:** `Mail_project_moritz` · Dieser Ordner ist ein isoliertes Experiment und ändert nichts an Stage 1/2/3 oder der App.

---

## TL;DR — Zusammenfassung in 30 Sekunden

- **Idee:** Aus der VIN (US-Pendant zur deutschen FIN) lassen sich über die kostenlose **NHTSA-API** zusätzliche Fahrzeugmerkmale (Kraftstoff, Hubraum, Zylinder, Leistung …) nachladen. Auslöser war der Vergleich mit *WirKaufenDeinAuto.de*, die deutlich mehr Felder abfragen als wir nutzen.
- **Datenlage:** Wir haben **alle 558.743 VINs lückenlos** (100 % vorhanden, 100 % im 17-Zeichen-Format, 98,5 % eindeutig). VIN-Decoding ist technisch problemlos möglich.
- **Abdeckung (validiert über 200 + 1.000 VINs):** **Hubraum ~99 %**, **Kraftstoff ~96 %**, **Zylinder ~88 %** sind sehr gut befüllt. **Leistung (PS) nur ~45 %**, **Getriebe ~12 %** → unbrauchbar.
- **Modelltests:** Die VIN-Features verbessern das Modell konsistent — +10,6 % gegen Baseline, +20,8 % gegen V2 (14k) und **+19,7 % gegen V2 auch bei 96k Zeilen**. Der Effekt bleibt von kleiner auf große Stichprobe stabil → **robustes, eigenständiges Signal** (siehe korrigierte Interpretation, Abschnitt 5).
- **Welches Merkmal trägt den Effekt? → Der Hubraum.** Per-Feature-Ablation: Hubraum allein bringt +19,8 %, praktisch identisch zu „alle drei" (+19,7 %). Kraftstoff (+3,3 %) und Zylinder (+11,9 % allein, aber redundant zum Hubraum) fügen darüber hinaus nichts hinzu. Für die Integration genügt **ein Merkmal: Hubraum** (zugleich mit 99 % die beste Abdeckung).
- **Empfehlung:** VIN-Integration **ernsthaft erwägen**. Einzige offene Frage: Bleiben die ~20 % auch beim vollen Datensatz (534k), wo die V2-Baseline ihr Bestniveau ($1.370) erreicht? Dafür der finale 534k-Test (Decode-Cache mit 100k VINs liegt bereits vor → nur noch ~450k nachzuladen).
- **Nebenbefund:** Neu- vs. Gebrauchtwagen können wir **nicht** unterscheiden — der Datensatz ist faktisch ein reiner Gebrauchtwagen-Auktionsdatensatz (nur 2,6 % Alter ≤ 0, Median 3 Jahre / 52.000 Meilen).

---

## 1. Idee und Motivation

### 1.1 Auslöser
Beim Gegentest auf *WirKaufenDeinAuto.de* (WKDA) fiel auf, dass dort für eine Preisschätzung **deutlich mehr Informationen** abgefragt werden als unser Modell nutzt — z. B. Kraftstoffart, Leistung, Vorbesitzer, Ausstattung, Scheckheft.

Frage an unser Projekt: **Welche dieser Felder könnten wir nachträglich gewinnen, ohne neue Erhebung?** Die Antwort: über die **VIN** (Vehicle Identification Number), die in jeder Zeile unseres Datensatzes steht. Sie ist das US-Pendant zur deutschen **FIN** (Fahrzeug-Identifikationsnummer) und codiert herstellerseitig technische Fahrzeugmerkmale.

### 1.2 Die NHTSA-API
Die **NHTSA vPIC API** (US-Verkehrsbehörde) dekodiert VINs **kostenlos und ohne API-Key**:

- Einzelabfrage: `https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/<VIN>?format=json`
- Batch (bis ~50 VINs/Request): `.../DecodeVINValuesBatch/` (POST, `data=VIN1;VIN2;...`)

Sie liefert u. a. Marke, Modell, Baujahr, **Kraftstoff, Hubraum, Zylinder, Leistung, Antrieb, Karosserie**.

> Wichtig: Die API ist für **US-VINs** ausgelegt — passt also exakt zu unserem Manheim-Datensatz. Für echte deutsche FINs wäre eine andere Quelle nötig.

### 1.3 Abgleich WKDA-Felder ↔ unser Datensatz

| WKDA fragt ab | Bei uns vorhanden? | Quelle |
|---|---|---|
| Marke / Modell / Variante | ✅ | `make`, `model`, `trim` |
| Erstzulassung (Alter) | ✅ | `vehicle_age` |
| Kilometerstand | ✅ | `odometer` |
| Getriebe | ✅ | `transmission` (Spalte) |
| Karosserie | ✅ | `body` |
| Außen-/Innenfarbe | ✅ | `color`, `interior` |
| Zustand | 🟡 nur Score 1–5 | `condition` |
| **Kraftstoffart** | ❌ Rohdaten / ✅ via VIN | NHTSA |
| **Leistung / Hubraum** | ❌ Rohdaten / 🟡 via VIN | NHTSA |
| Vorbesitzer, Scheckheft, Ausstattung, TÜV | ❌ nicht beschaffbar | — |

---

## 2. Nebenbefund: Neu vs. Gebraucht

Geprüft, ob unser Modell Neu- und Gebrauchtwagen unterscheiden könnte. **Nein** — der Datensatz ist faktisch ein reiner Gebrauchtwagen-Großhandels-Auktionsdatensatz (Manheim):

| Kennzahl | Wert |
|---|---|
| Anteil „quasi-neu" (Alter ≤ 0) | 2,6 % |
| Anteil < 1.000 Meilen | 0,5 % |
| Median Alter / Laufleistung | 3 Jahre / 52.254 Meilen |

Eine Neu/Gebraucht-Klassifikation würde Neuwagen-Listenpreise als zusätzliche Datenquelle erfordern. Das Modell erfasst „fast neu" nur indirekt über `vehicle_age=0` + niedriger `odometer` + hohe `condition`.

---

## 3. Datenabdeckung der VIN

### 3.1 Sind die VINs überhaupt verwertbar?
Geprüft auf dem vollen Datensatz (558.743 Zeilen):

| Prüfung | Ergebnis |
|---|---|
| VIN fehlend (NaN) | **0** |
| Länge exakt 17 Zeichen | **100,0 %** |
| Eindeutige VINs | **550.245 / 558.743 = 98,5 %** |
| Plausible Präfixe | ✅ (WBA = BMW, YV1 = Volvo, 5XY = Kia …) |

→ VIN-Decoding ist ohne Vorfilterung möglich.

### 3.2 Trefferquote der dekodierten Felder
Zwei unabhängige Stichproben (`01_coverage_test.py`), um die Stabilität zu prüfen:

| Feld | 200 VINs | 1.000 VINs | Bewertung |
|---|---|---|---|
| BodyClass | 100,0 % | 100,0 % | ✅ (haben wir aber schon als `body`) |
| **DisplacementL** (Hubraum) | 98,0 % | **99,1 %** | ✅ exzellent |
| **FuelTypePrimary** (Kraftstoff) | 96,5 % | **95,7 %** | ✅ exzellent |
| **EngineCylinders** (Zylinder) | 84,5 % | **87,7 %** | ✅ gut |
| DriveType (Antrieb) | 63,5 % | 62,8 % | 🟡 mittel |
| EngineHP (Leistung) | 44,5 % | 46,7 % | 🟡 lückenhaft |
| TransmissionStyle | 10,5 % | 13,4 % | ❌ unbrauchbar |

**Interpretation:** Die Quoten sind über beide Stichproben praktisch identisch (Abweichung < 3 Prozentpunkte = Rauschen) → die Messung ist stabil. **Nutzbare neue Features: Kraftstoff, Hubraum, Zylinder.** Leistung und Getriebe sind zu lückenhaft (Getriebe haben wir ohnehin als eigene Spalte).

---

## 4. Modelltests (Ablationen)

Methodik aller Ablationen: identischer Train/Test-Split (80/20, `random_state=42`),
identisches Modell (`HistGradientBoostingRegressor`, log1p-Ziel, OrdinalEncoder),
**nur das Feature-Set unterscheidet sich** zwischen Arm A und B. So misst die
MAE-Differenz ausschließlich den Beitrag der VIN-Features.

> Hinweis: Wir verwenden HistGB für beide Arme. Der Vergleich ist dadurch intern
> valide; die absoluten MAE-Werte sind nicht 1:1 mit dem produktiven
> XGBoost-Ensemble (V2, MAE $1.370 auf vollem Datensatz) vergleichbar.

### Test 3 — VIN vs. einfache Baseline (`02_ablation_vs_baseline.py`, 12.000 Zeilen)

| Modell | MAE | R² |
|---|---|---|
| A) Baseline (make, model, body, condition, odometer, vehicle_age, year_month, sale_month) | $1.994 | 0,8577 |
| B) + VIN (Kraftstoff, Hubraum, Zylinder) | **$1.783** | **0,9024** |
| **Differenz** | **−$211 (+10,6 %)** | +0,045 |

### Test 4 — VIN zusätzlich zu V2 (`03_ablation_vs_v2.py`, 14.000 Zeilen → 13.378 verwertbar)

| Modell | MAE | R² |
|---|---|---|
| A) V2-Features (model_year, vehicle_age, odometer, condition, trim, transmission, state, color, interior, make_model) | $2.368 | 0,7545 |
| B) V2 + VIN (Kraftstoff, Hubraum, Zylinder) | **$1.876** | **0,8407** |
| **Differenz** | **−$492 (+20,8 %)** | +0,086 |

### Test 5 — VIN zusätzlich zu V2, GROSSE Stichprobe (`test_vin_ablation_v2_100k.py`, 100.000 Zeilen → 95.868 verwertbar)

Dieser Test sollte klären, ob der Effekt aus Test 4 nur ein Artefakt der kleinen Stichprobe ist.

| Modell | MAE | R² |
|---|---|---|
| A) V2-Features (inkl. trim) | $2.076 | 0,8303 |
| B) V2 + VIN (Kraftstoff, Hubraum, Zylinder) | **$1.668** | **0,8909** |
| **Differenz** | **−$408 (+19,7 %)** | +0,061 |

VIN-Abdeckung @ 100k: Kraftstoff 96,1 %, Hubraum 99,2 %, Zylinder 88,1 % — identisch zu den kleinen Stichproben.

### Test 6 — Per-Feature-Ablation: welches Merkmal trägt den Effekt? (`05_per_feature_ablation.py`, 100.000 Zeilen → 95.868 verwertbar)

Bis hierhin kamen Kraftstoff, Hubraum und Zylinder immer als Dreierpaket dazu. Dieser Test schaltet jedes Merkmal **einzeln** zu, um den isolierten Beitrag zu messen.

| Arm | MAE | R² | Verbesserung vs. Basis A |
|---|---|---|---|
| A) V2 (Basis) | $2.076 | 0,8303 | — |
| B) V2 + Kraftstoff | $2.007 | 0,8370 | +3,3 % |
| **C) V2 + Hubraum** | **$1.665** | **0,8899** | **+19,8 %** |
| D) V2 + Zylinder | $1.828 | 0,8771 | +11,9 % |
| E) V2 + alle drei | $1.668 | 0,8909 | +19,7 % |

**Ergebnis: Der gesamte Effekt steckt im Hubraum.** Hubraum allein (+19,8 %) ist praktisch identisch zu „alle drei" (+19,7 %) — „alle drei" ist sogar minimal schlechter ($1.668 vs. $1.665, Rauschen).

- **Kraftstoff** trägt für sich kaum bei (+3,3 %); die meisten Fahrzeuge sind ohnehin „Gasoline".
- **Zylinder** sieht allein brauchbar aus (+11,9 %), aber das ist nur die Korrelation mit dem Hubraum (mehr Zylinder = größerer Motor). **Sobald Hubraum vorhanden ist, fügt Zylinder nichts mehr hinzu** → redundant.
- **Inhaltlich plausibel:** Der Hubraum ist ein direkter, kontinuierlicher Proxy für Motorgröße/Leistungsklasse und unterscheidet zwei sonst identische Fahrzeuge (gleicher Trim, gleiches Modell) im Preis. Kraftstoff und Zylinder liefern davon nur eine vergröberte Teilinfo.

**Konsequenz für die Integration:** Es genügt **ein einziges Merkmal — der Hubraum** — das zugleich mit 99 % die beste Abdeckung hat. Kraftstoff und Zylinder können entfallen, ohne Genauigkeit zu verlieren. Das ist auch für das Paper die stärkere, sparsamere Aussage (Parsimonie).

---

## 5. Interpretation — bitte unbedingt lesen

> **Hinweis:** Eine frühere Fassung dieses Abschnitts vermutete, der VIN-Effekt sei
> nur ein Artefakt der kleinen Stichprobe und würde bei mehr Daten verschwinden.
> Test 5 (100k) hat diese Hypothese **getestet und widerlegt**. Die folgende
> Fassung ist die korrigierte Interpretation.

### 5.1 Die ursprüngliche Skepsis — und warum sie falsch war
Vermutung war: Auf wenig Daten kann das Modell die Motorisierung nicht aus
`make`/`model`/`trim` lernen, daher hilft ein explizites Signal wie „Hubraum 3.5 L"
überproportional; mit mehr Daten sollte der VIN-Nutzen **schrumpfen**.

**Das ist nicht eingetreten.** Der Zusatznutzen blieb über alle Stichprobengrößen
nahezu konstant:

| Test | Stichprobe | VIN-Zusatznutzen (vs. V2) |
|---|---|---|
| Test 4 | 14k | +20,8 % |
| Test 5 | 96k | +19,7 % |

Von 14k auf 96k Zeilen — also fast Versiebenfachung der Daten — sinkt der Effekt
nur um gut 1 Prozentpunkt. Das ist **kein Stichproben-Artefakt**, sondern ein
robustes, eigenständiges Signal.

### 5.2 Warum tragen Kraftstoff/Hubraum/Zylinder echten Mehrwert?
`trim` codiert die Motorisierung **unvollständig und inkonsistent**: Trim-Namen
sind frei vergeben („LX", „Limited", „Sport") und sagen oft nichts über Hubraum
oder Kraftstoff. Ein 2.0-L-Benziner und ein 3.5-L-Benziner können denselben Trim
tragen. Die VIN dagegen codiert die technische Motorvariante **eindeutig und
standardisiert** — genau die Information, die im Trim fehlt.

### 5.3 Die eine verbleibende Einschränkung
Die V2-Baseline liegt bei 96k bei $2.076; das produktive V2 erreicht auf den
vollen **534k** Zeilen aber $1.370. Bei 96k sind die hochkardinalen Features
(`trim`, `make_model`, `state`) also noch nicht voll ausgereizt. Offen bleibt
daher nur: Bleiben die ~20 %, wenn die V2-Baseline ihr Bestniveau erreicht, oder
schließt sich ein Teil der Lücke? Der stabile Trend 14k→96k spricht dafür, dass
ein erheblicher Teil bestehen bleibt — die exakte Zahl liefert erst der 534k-Test.

### 5.4 Schlussfolgerung
- **Belegt:** Kraftstoff/Hubraum/Zylinder aus der VIN sind sehr gut verfügbar
  (>88 %) und tragen **robustes, eigenständiges Signal**, das über `trim` hinausgeht
  — konsistent über 14k und 96k Zeilen.
- **Offen:** der exakte Mehrwert beim vollen Datensatz (534k); erwartbar weiterhin
  spürbar, evtl. etwas unter 20 %.

---

## 6. Empfehlung

1. VIN-Integration **ernsthaft erwägen** — der Mehrwert ist über zwei
   Stichprobengrößen robust belegt (~20 %). **Es genügt der Hubraum** als
   einziges neues Merkmal (Test 6); Kraftstoff/Zylinder sind verzichtbar.
2. **Finalen 534k-Test** durchführen für die zitierfähige Zahl. Der Decode-Cache
   mit 100k VINs liegt bereits vor (`vin_decoded_cache.csv`) → nur noch ~450k
   VINs nachzuladen statt aller 550k.
3. Für das Paper: Abschnitt zur VIN-Anreicherung mit dem **methodischen Lerneffekt**
   — anfängliche Skepsis (Stichproben-Confounder vermutet), durch den 100k-Test
   widerlegt. Das zeigt sauberes wissenschaftliches Vorgehen.
4. Mengengerüst Vollausbau: ~450k verbleibende VINs ÷ 50/Batch ≈ 9.000 Requests,
   grob **~2,5 Stunden** → einmaliger Offline-Batch mit Caching, **kein**
   Live-Feature in der App.

---

## 7. Reproduktion

```bash
# Abhängigkeit (nur requests zusätzlich zum Projekt-Stack)
uv run python experiments/vin_fin_enrichment/01_coverage_test.py        # Abdeckung (SAMPLE_SIZE anpassbar)
uv run python experiments/vin_fin_enrichment/02_ablation_vs_baseline.py  # Test 3
uv run python experiments/vin_fin_enrichment/03_ablation_vs_v2.py        # Test 4
```

- Alle Skripte lesen `car_prices_clean.csv` **read-only** und schreiben nur in
  diesen Ordner. Sie ändern **nichts** an Modellen, Pipeline oder App.
- Internet nötig (NHTSA-API). `random_state=42` macht die Stichproben reproduzierbar.

## 8. Dateien in diesem Ordner

| Datei | Inhalt |
|---|---|
| `FIN_Test.md` | Diese Dokumentation |
| `01_coverage_test.py` | Test 1+2: VIN-Feld-Abdeckung (200 / 1.000) |
| `02_ablation_vs_baseline.py` | Test 3: VIN vs. einfache Baseline |
| `03_ablation_vs_v2.py` | Test 4: VIN zusätzlich zu V2-Features (14k) |
| `04_ablation_vs_v2_100k.py` | Test 5: VIN zusätzlich zu V2-Features (100k, mit inkrementellem Cache) |
| `05_per_feature_ablation.py` | Test 6: Beitrag je Merkmal einzeln (nutzt Cache, keine API) |
| `vin_decoded_cache.csv` | 99.703 bereits dekodierte VINs (Kraftstoff/Hubraum/Zylinder/Leistung/Antrieb), per VIN joinbar — Basis für Test 6 und den 534k-Test |

## 9. Offene Punkte / Grenzen

- Nur **ein** Split je Ablation → kleine Effekte sind verrauscht; sauberer wäre
  Cross-Validation.
- Leistung (PS) per VIN zu lückenhaft (~45 %); Imputation würde Aussagekraft verwässern.
- Keine Prüfung, ob NHTSA-Felder über die Zeit/Hersteller systematisch fehlen
  (z. B. bestimmte Marken schlechter abgedeckt).
- **Finaler 534k-Test steht aus** — er liefert die zitierfähige Zahl gegen die
  voll ausgereizte V2-Baseline ($1.370). Vorarbeit: `vin_decoded_cache.csv` mit
  100k VINs liegt bereits vor.
