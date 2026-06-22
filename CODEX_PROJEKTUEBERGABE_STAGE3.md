# Projektübergabe für einen neuen Codex-Chat

Stand: 22.06.2026

## Zweck dieser Datei

Diese Datei dokumentiert alle wesentlichen Arbeiten, die Codex in der bisherigen
Sitzung am Projekt durchgeführt hat. Ein neuer Chat soll damit:

1. den Projektstand schnell verstehen,
2. die fachlichen Entscheidungen von Stage 3 nachvollziehen,
3. Stage 3 bei fehlenden Dateien vollständig reproduzieren,
4. die vorhandenen Ergebnisse überprüfen und weiterentwickeln können.

Wichtiger Hinweis: Die unten beschriebenen Änderungen sind im aktuellen lokalen
Arbeitsordner vorhanden, aber zum Zeitpunkt dieser Übergabe noch **nicht committed
oder zu GitHub gepusht**.

## Projektkontext

Repository:

`https://github.com/digital-business-lectures/ai-project-mail.git`

Lokaler Arbeitsordner:

`C:\Users\xariv\Documents\GitHub\ai-project-mail`

Das Projekt entwickelt einen hybriden KI-Agenten für dynamische
Gebrauchtwagenpreise:

- Stage 1: ML-Modell für den Fahrzeugwert
- Stage 2: makroökonomische CPI-Anpassung
- Stage 3: konservative saisonale Anpassung nach Karosserieform und Monat
- geplantes Endprodukt: Streamlit-Demo und später LLM-Orchestrierung

Als erste Kontextquelle wurde außerdem diese Präsentations-Sprechtextdatei
gelesen:

`C:\Users\xariv\Downloads\zwischenpraesi_sprechtext_nach_cleaning.md`

Die zentralen bestehenden Projektdateien waren außerdem `Aktueller_Stand.md`,
`README.md`, das Stage-1-Training, die Stage-2-Logik und die Streamlit-App.

## Ausgangslage vor Stage 3

Vor den Änderungen waren Stage 1 und Stage 2 implementiert:

- Datensatz: `car_prices_features.csv`
- rund 534.000 verwertbare Auktionsverkäufe aus 2014 und 2015
- gespeichertes Modell: `models/price_model.joblib`
- Modelltyp: `HistGradientBoostingRegressor`
- Stage-1-Test-MAE: ungefähr 1.850 USD
- Stage-1-R²: ungefähr 0,882
- Stage 2 verwendet `macro_index.csv` und den CPI für gebrauchte Fahrzeuge
- CPI-Referenz ist der Jahresdurchschnitt 2015 mit Faktor 1,0

Stage 3 war als saisonale Regel nach Karosserieform und Verkaufsmonat geplant,
aber noch nicht umgesetzt.

## Zuerst erstellte Stage-3-Variante und warum sie verworfen wurde

Die erste Stage-3-Version verglich rohe monatliche Medianpreise innerhalb einer
Karosserieform:

```text
raw_factor = median_price(body, month) / median_price(body)
```

Kleine Gruppen wurden Richtung 1,0 geglättet und Faktoren auf 0,85 bis 1,15
begrenzt. Diese Variante wurde anschließend kritisch überprüft und verworfen.

Gefundene Probleme:

- Der Fahrzeugmix unterscheidet sich je Monat. Teurere, jüngere oder besser
  erhaltene Fahrzeuge wurden dadurch fälschlich als Saisoneffekt interpretiert.
- Stage 1 erhielt in der App bereits den ausgewählten Zielmonat. Stage 3
  multiplizierte anschließend noch einmal einen Monatsfaktor. Der Monatseffekt
  wurde damit potenziell doppelt berücksichtigt.
- Der Datensatz enthält nur die Monate Januar bis Juli und Dezember.
- August bis November fehlen vollständig.
- April und Juli haben teilweise sehr wenige Beobachtungen. Die ersten Faktoren
  zeigten deshalb unplausible Ausschläge bis zur Grenze von plus/minus 15 Prozent.

Diese erste Ergebnisversion darf nicht wiederhergestellt oder als fachliches
Resultat verwendet werden.

## Aktuelle, korrigierte Stage-3-Methode

Die aktuelle Methode kontrolliert den Fahrzeugmix über das bestehende
Stage-1-Modell und entfernt das historische CPI-Preisniveau.

### Feste Stage-1-Referenz

Für alle Stage-1-Vorhersagen wird unabhängig vom Zielmonat dieselbe historische
Marktreferenz verwendet:

```text
REFERENCE_YEAR_MONTH = "2015-02"
REFERENCE_MONTH = 2
```

Fahrzeugalter, Kilometerstand, Zustand, Marke, Modell und Karosserieform bleiben
fahrzeugspezifisch. Nur `sale_month` und `year_month` werden für die
Stage-1-Baseline auf Februar 2015 gesetzt.

Dadurch hat Stage 1 folgende Bedeutung:

> Wert dieses Fahrzeugs unter dem Preisniveau und der Monatsreferenz Februar 2015.

Stage 2 hebt diese Baseline anschließend mit dem CPI auf das gewählte Zieldatum.
Stage 3 fügt erst danach den karosseriespezifischen Monatseffekt hinzu.

### Berechnung der Faktoren

Für jede historische Verkaufszeile:

```text
normalized_price = sellingprice / cpi_multiplier(year_month)
price_ratio = normalized_price / stage1_prediction_at_2015_02
```

Danach wird innerhalb jeder Karosserieform normalisiert:

```text
relative_ratio = price_ratio / median(price_ratio for body)
raw_factor(body, month) = median(relative_ratio for body and month)
```

Der rohe Faktor wird mit einer Prior-Stärke von 1.000 Beobachtungen Richtung 1,0
gedämpft:

```text
weight = observations / (observations + 1000)
seasonal_factor = 1 + (raw_factor - 1) * weight
seasonal_factor = clip(seasonal_factor, 0.85, 1.15)
```

Regeln zur Datenqualität:

- Fehlende Monate erhalten Faktor 1,0.
- Monate ohne Daten werden als `no_data` gekennzeichnet.
- Unter 100 Beobachtungen ist die Sicherheit `low`.
- Zwischen 100 und 999 Beobachtungen ist sie `medium`.
- Ab 1.000 Beobachtungen ist sie `high`.
- Ein Monat wird nur ab 100 Beobachtungen als bester oder schwächster Monat
  empfohlen, sofern eine Karosserieform überhaupt so viele Daten besitzt.

Finale Preisformel:

```text
stage2_price = stage1_reference_price * cpi_multiplier(target_year_month)
final_price = stage2_price * seasonal_factor(body, target_month)
```

## Datenabdeckung

Historisch beobachtete Monate:

```text
1, 2, 3, 4, 5, 6, 7, 12
```

Nicht beobachtete Monate:

```text
8, 9, 10, 11
```

August bis November müssen neutral bei Faktor 1,0 bleiben. Für diese Monate darf
die Oberfläche keine historische Saisonwirkung behaupten.

Die Daten stammen fast vollständig aus Dezember 2014 bis Juli 2015. Stage 3 ist
daher ausdrücklich eine konservative Heuristik und kein kausaler oder über viele
Jahre replizierter Nachweis von Saisonalität.

## Neu erstellte Dateien

### `scripts/stage3_seasonality.py`

Enthält die komplette Stage-3-Logik:

- `prepare_seasonality_data()`
- `calculate_seasonality_factors()`
- `build_seasonality_factors()`
- `load_seasonality_factors()`
- `get_seasonality_row()`
- `get_seasonal_factor()`
- `apply_stage3()`

Wichtige Konstanten:

- `REFERENCE_YEAR_MONTH = "2015-02"`
- `REFERENCE_MONTH = 2`
- `FACTOR_MIN = 0.85`
- `FACTOR_MAX = 1.15`
- `SHRINKAGE_OBSERVATIONS = 1000`
- `MIN_RECOMMENDATION_OBSERVATIONS = 100`

### `scripts/evaluate_stage3.py`

Erzeugt die Faktoren, die Zusammenfassung und eine getrennte 80/20-Prüfung der
Saisonregel.

Bei der Prüfung werden die Stage-3-Faktoren nur aus zufällig ausgewählten 80
Prozent der aufbereiteten Zeilen abgeleitet und auf den übrigen 20 Prozent
angewendet. Dies prüft die Saisonregel, ist aber kein vollständig unabhängiger
neuer Stage-1-Modelltest, weil das gespeicherte Stage-1-Modell separat zuvor
trainiert wurde.

### Generierte Ergebnisse

- `models/seasonality_factors.csv`
- `models/stage3_evaluation.json`
- `model_results_stage3.md`

`seasonality_factors.csv` enthält 540 Zeilen:

- 45 Karosserieformen
- jeweils 12 Monate
- Faktor, Effekt in Prozent, Beobachtungszahl und Sicherheitsklasse
- bester und schwächster ausreichend beobachteter Monat

## Aktuelle Stage-3-Ergebnisse

Verwertbare historische Verkäufe:

```text
529.790
```

80/20-Prüfung:

| Kennzahl | Ergebnis |
|---|---:|
| Regel-Trainingszeilen | 423.984 |
| Prüfzeilen | 105.806 |
| MAE ohne Stage 3 | 1.895,03 USD |
| MAE mit Stage 3 | 1.870,20 USD |
| absolute Änderung | -24,82 USD |
| relative Änderung | -1,31 % |

Beispiele für große Segmente:

| Karosserie | stärkster Monat | Faktor | schwächster Monat | Faktor |
|---|---|---:|---|---:|
| sedan | März | 1,0381 | Dezember | 0,9286 |
| suv | März | 1,0290 | Dezember | 0,9158 |
| hatchback | Januar | 1,0292 | Juni | 0,9506 |
| minivan | März | 1,0149 | Dezember | 0,9357 |
| coupe | März | 1,0220 | Dezember | 0,9514 |
| convertible | März | 1,0197 | Dezember | 0,9487 |

Diese Faktoren sind deutlich kleiner und plausibler als die verworfenen rohen
Medianeffekte.

## Änderungen an der Streamlit-App

Geänderte Datei:

`app/streamlit_app.py`

Umgesetzt wurde:

- Import und Laden von `stage3_seasonality`
- Stage-1-Eingabe verwendet immer die feste Referenz Februar 2015
- Berechnung von Stage 1, Stage 2 und finalem Stage-3-Preis
- Hauptkennzahl zeigt den finalen Preis aus Markt und Saison
- separate Anzeigen für:
  - Stage-1-Baseline
  - Stage-2-Marktpreis
  - Saisonfaktor
  - CPI-Multiplikator
  - besten historisch beobachteten Verkaufsmonat
- verständliche Hinweise für starke, schwache und neutrale Monate
- eigener Hinweis bei Monaten ohne historische Daten
- Tabelle mit allen zwölf Saisonmonaten der gewählten Karosserieform
- Erklärung aller drei Stufen im Bereich „Was passiert hier?“
- veraltete Streamlit-Option `use_container_width=True` durch
  `width="stretch"` ersetzt

Wichtiger Testfall:

- Zielmonat August
- erwarteter Faktor: `1.0000`
- erwarteter Hinweis: keine historischen Verkäufe, daher neutrale Anpassung

## Weitere geänderte Dateien

### `README.md`

- Stage 3 in Architektur und Dateistruktur ergänzt
- Evaluationsbefehl ergänzt
- korrigierte, residualbasierte Methode dokumentiert

### `Aktueller_Stand.md`

- Stage 3 von offen auf fertig gesetzt
- neue Dateien und Befehle ergänzt
- nächste Aufgaben neu priorisiert
- Stage-3-Methode als bereinigte Modellabweichung beschrieben

### `pyproject.toml` und `uv.lock`

`scikit-learn` wurde auf exakt Version 1.8.0 festgelegt:

```toml
"scikit-learn==1.8.0"
```

Grund: `models/price_model.joblib` wurde mit scikit-learn 1.8.0 gespeichert und
konnte unter 1.9.0 nicht zuverlässig geladen werden.

### `.gitignore`

`.uv-cache/` wurde ergänzt, weil bei der lokalen Einrichtung ein projektnaher
uv-Cache verwendet wurde.

## Reproduktion in einem neuen Chat oder Klon

### Voraussetzung

Folgende bereits bestehende Dateien müssen vorhanden sein:

- `car_prices_features.csv`
- `macro_index.csv`
- `models/price_model.joblib`
- `models/price_model_metrics.json`
- Stage-1- und Stage-2-Skripte

Die Python-Umgebung muss scikit-learn 1.8.0 verwenden.

### Standardablauf

Vom Projektwurzelverzeichnis:

```powershell
uv sync
uv run python scripts/evaluate_stage3.py
uv run streamlit run app/streamlit_app.py
```

Die Evaluation muss folgende Dateien neu erzeugen:

```text
models/seasonality_factors.csv
models/stage3_evaluation.json
model_results_stage3.md
```

### Erwartete Prüfungen

Ein neuer Chat soll mindestens prüfen:

1. Python-Syntax von App und beiden Stage-3-Skripten.
2. Genau 12 Faktorzeilen je Karosserieform.
3. Keine doppelten Kombinationen aus `body` und `sale_month`.
4. Alle Faktoren zwischen 0,85 und 1,15.
5. Alle Monate ohne Beobachtung haben exakt Faktor 1,0.
6. Beobachtete Monate sind genau 1 bis 7 und 12.
7. August bis November sind als unbeobachtet dokumentiert.
8. Der 80/20-Prüf-MAE mit Stage 3 ist nicht schlechter als ohne Stage 3.
9. Die Streamlit-App startet ohne Ausnahme.
10. August zeigt Faktor 1,0000 und den Hinweis auf fehlende Daten.

Zuletzt erfolgreich geprüft:

- Python-Kompilierung erfolgreich
- UTF-8-Prüfung erfolgreich
- 540 Faktorzeilen und 45 Karosserieformen
- alle Struktur- und Wertebedingungen erfüllt
- Streamlit AppTest für Standardansicht erfolgreich
- Streamlit AppTest für August erfolgreich
- lokaler HTTP-Status der App: 200

## Aktueller Git-Status

Zum Zeitpunkt der Übergabe waren folgende Änderungen noch nicht committed:

```text
M  .gitignore
M  Aktueller_Stand.md
M  README.md
M  app/streamlit_app.py
M  pyproject.toml
M  uv.lock
?? CODEX_PROJEKTUEBERGABE_STAGE3.md
?? model_results_stage3.md
?? models/seasonality_factors.csv
?? models/stage3_evaluation.json
?? scripts/evaluate_stage3.py
?? scripts/stage3_seasonality.py
```

Ein neuer Klon von GitHub enthält diese Änderungen erst, nachdem sie committed
und gepusht wurden. Wenn die Dateien im neuen Arbeitsordner fehlen, soll Stage 3
anhand dieser Übergabe rekonstruiert werden.

## Diagnose der GitHub-/Codex-Verbindung

Während der Sitzung meldete Codex beim erneuten Importieren des Repositories:

> The repository does not seem to exist anymore. You may not have access, or it
> may have been deleted or renamed.

Direkte Prüfung außerhalb der Codex-Sandbox ergab:

- Repository existiert.
- Remote-URL ist korrekt.
- GitHub-Konto `Paelus` hat Zugriff.
- `git ls-remote origin` funktioniert.
- `git fetch origin --prune` funktioniert.

Die Fehlermeldung betrifft daher die separate GitHub-App-/Connector-Autorisierung
von Codex, nicht den lokalen Git-Zugriff. Für einen neuen Codex-Import muss die
OpenAI-/Codex-GitHub-App Zugriff auf die Organisation `digital-business-lectures`
und speziell auf `ai-project-mail` besitzen. Möglicherweise muss ein
Organisationsadministrator diese App genehmigen.

## Empfohlene nächste Projektschritte

Nach Stage 3 wurden diese Aufgaben priorisiert:

1. Preisbereich statt nur eines einzelnen Schätzwerts anzeigen, vorzugsweise
   basierend auf dem MAE des jeweiligen Preissegments.
2. Stage-1-Modell auf dem vollständigen Datensatz trainieren.
3. LLM-Orchestrierung für eine verständliche Erklärung der Stage-1- bis
   Stage-3-Ergebnisse ergänzen.
4. Ergebnisse und Architektur in das LNCS-Paper übernehmen.

## Auftrag an einen neuen Codex-Chat

Empfohlener Startprompt zusammen mit dieser Datei:

> Lies `CODEX_PROJEKTUEBERGABE_STAGE3.md` vollständig. Prüfe anschließend, welche
> der beschriebenen Stage-3-Dateien im aktuellen Arbeitsordner vorhanden sind.
> Rekonstruiere fehlende Teile exakt nach der dokumentierten korrigierten Methode,
> führe die Evaluation erneut aus und vergleiche die Ergebnisse mit den hier
> angegebenen Referenzwerten. Verwende nicht die verworfene Methode aus rohen
> monatlichen Medianpreisen. Verändere keine vorhandenen Nutzeränderungen und
> berichte Abweichungen transparent.
