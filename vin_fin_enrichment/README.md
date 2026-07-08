# FIN/VIN Enrichment

Dieser Ordner enthält die aktuelle technische Grundlage für die FIN/VIN-
Anreicherung des Stage-1-Modells. Ziel ist es, aus der US-VIN zusätzliche
Fahrzeugmerkmale über die NHTSA-vPIC-API zu gewinnen. Für das finale
Stage-1-Modell ist vor allem der **Hubraum** (`displacement`) relevant.

## Aktuelle Dateien

| Datei | Zweck |
|---|---|
| `build_full_vin_cache.py` | dekodiert alle eindeutigen VINs und erzeugt den vollständigen Decode-Cache |
| `build_micro_fin_dataset.py` | verbindet den Decode-Cache mit `car_prices_clean.csv` und erzeugt `car_prices_fin.csv` |

## Reproduktion

```bash
uv run python vin_fin_enrichment/build_full_vin_cache.py
uv run python vin_fin_enrichment/build_micro_fin_dataset.py
```

Der vollständige Decode-Cache kann aus der NHTSA-API reproduziert werden und ist
deshalb nicht als aktuelles Kerndokument gedacht. Alte Ablations- und
Zwischenstandsskripte sowie ein alter Sample-Cache liegen im Archiv unter
`archive/fin_enrichment_legacy_2026-07-08/`.
