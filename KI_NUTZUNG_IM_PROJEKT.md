# KI-Nutzung im Projekt

Im Projekt wurden KI-Tools unterstützend eingesetzt, allerdings nicht als Ersatz
für eigene fachliche Entscheidungen. Die zentrale Projektlogik, die Bewertung
der Ergebnisse und die finale Einordnung der Stages wurden vom Team getroffen.
KI wurde vor allem genutzt, um Ideen schneller umzusetzen, Varianten zu prüfen
und technische Arbeitsschritte effizienter auszuführen.

## Eingesetzte KI-Tools

### ChatGPT / Codex

Codex wurde vor allem als technischer Assistent eingesetzt. Die Nutzung lag
schwerpunktmäßig bei:

- Unterstützung beim Schreiben und Strukturieren von Python-Code,
- Debugging von Fehlermeldungen,
- Refactoring und Aufräumen der Repository-Struktur,
- Formulieren von Dokumentationsentwürfen,
- Erstellen erster Textentwürfe für README, Projektstand und Hausarbeit,
- Weiterentwicklung der Streamlit-Oberfläche,
- Überprüfen, ob Ergebnisse logisch zu Stage 1, Stage 2 oder Stage 3 gehören.

Wichtig ist: Codex hat dabei vor allem Vorschläge gemacht und Codeänderungen
ausgeführt. Die fachliche Entscheidung, welche Änderungen sinnvoll sind und wie
sie in den Projektplan passen, wurde durch das Team getroffen. Beispielsweise
wurde nachträglich bewusst entschieden, das neue V2-Modell nicht als Stage-3-
Ergebnis zu interpretieren, sondern korrekt Stage 1 zuzuordnen.

### ChatGPT als Schreib- und Reflexionshilfe

Neben der technischen Umsetzung wurde ChatGPT zur sprachlichen Unterstützung
genutzt, etwa um:

- komplizierte Modell- und Evaluationsergebnisse verständlicher zu erklären,
- mögliche Rückfragen vorzubereiten,
- erste Formulierungen für wissenschaftliche Abschnitte zu skizzieren,
- Argumentationslinien für Einleitung, Theorie und Diskussion zu entwickeln.

Diese Texte wurden nicht ungeprüft übernommen. Sie dienten als Ausgangspunkt,
der anschließend fachlich geprüft, gekürzt, angepasst oder verworfen wurde.

## Eigene Leistungen und Entscheidungen

Die wesentlichen Projektentscheidungen lagen beim Team. Dazu gehörten unter
anderem:

- Festlegung der Drei-Stufen-Architektur,
- Entscheidung, Stage 1, Stage 2 und Stage 3 fachlich sauber zu trennen,
- Bewertung, dass die große V2-Verbesserung zu Stage 1 gehört,
- Auswahl und Interpretation der relevanten Metriken,
- Entscheidung, MMR, VIN und Seller wegen möglichem Leakage nicht als Features
  zu verwenden,
- kritische Einordnung der Stage-2- und Stage-3-Ergebnisse,
- Gestaltung der Streamlit-App aus Nutzersicht,
- Prüfung, welche Informationen in der Oberfläche sichtbar sein sollen und
  welche nur als Entwicklerdetails dienen.

## Rolle der KI im Arbeitsprozess

KI wurde im Projekt vor allem als beschleunigendes Werkzeug genutzt: Sie half
dabei, technische Umsetzungsschritte schneller durchzuführen, Fehlerquellen zu
finden und Textentwürfe vorzubereiten. Die Arbeit bestand aber nicht darin,
KI-Ausgaben einfach zu übernehmen. Vielmehr wurden die Vorschläge laufend mit
dem Projektplan, den Daten, den Modellergebnissen und der eigenen fachlichen
Einschätzung abgeglichen.

Kurz gesagt: Die KI war im Projekt eher eine ausführende und unterstützende
Arbeitskraft. Die Richtung, Bewertung und finale Verantwortung lagen beim Team.

---

## Ergänzung: Vollständige Auflistung der eingesetzten KI-Tools

Zur Transparenz nennen wir alle tatsächlich genutzten Werkzeuge und wofür:

- **ChatGPT / Codex (OpenAI):** technische Code-Unterstützung, Debugging,
  Repository-Refactoring, erste Textentwürfe für Doku und Hausarbeit.
- **Claude / Claude Code (Anthropic):** eingesetzt für die datengetriebenen
  Experimente rund um die FIN-Anreicherung — u. a. das Schreiben der
  Abdeckungs- und Ablationsskripte, das Ausführen der Tests, das Erstellen der
  Präsentationsfolien und der zugehörigen Dokumentation.

Beide Tools wurden als **Assistenten** genutzt. Die Fragestellungen, die
Versuchsplanung, die Bewertung der Ergebnisse und die daraus abgeleiteten
Entscheidungen kamen vom Team.

## Ergänzung: Konkrete Belege für eigenes Verständnis und Kontrolle

Damit nachvollziehbar ist, dass wir die KI-Ausgaben nicht ungeprüft übernommen,
sondern fachlich verstanden und kontrolliert haben, hier konkrete Beispiele:

- **Lamborghini-Fehlprognose erkannt und diagnostiziert:** Beim Testen der App
  fiel uns auf, dass ein Lamborghini Gallardo nur ~8.300 $ vorhergesagt bekam.
  Wir haben die Ursachen selbst hergeleitet: widersprüchliche Eingabe
  (Karosserie „g sedan"), nur 4 Lamborghinis im Datensatz und die generelle
  Schwäche des Modells im Luxussegment. Daraus folgte die bewusste Einordnung
  als Datenabdeckungs-Grenze und ein To-Do zur Eingabe-Plausibilisierung.
- **Data Leakage aktiv vermieden:** Wir haben entschieden, `MMR`, `VIN` und
  `seller` nicht als Modell-Features zu verwenden, weil sie das Ergebnis
  verfälschen würden (MMR ist bereits eine Preisschätzung; VIN/seller sind
  Kennungen). Das ist eine fachliche, keine von der KI vorgegebene Entscheidung.
- **Statistischen Trugschluss erkannt und gegengeprüft:** Eine erste
  FIN-Ablation auf kleiner Stichprobe zeigte ~20 % Verbesserung. Wir haben
  erkannt, dass dieser Wert durch Stichprobengröße und Merkmals-Kardinalität
  verzerrt sein kann, und ihn auf dem **vollen Datensatz** gegengeprüft
  (Ergebnis: belastbare ~12–13 %). Die anfängliche Interpretation wurde
  daraufhin bewusst korrigiert.
- **Ursache des Effekts isoliert:** Per Einzel-Feature-Ablation haben wir
  festgestellt, dass von den FIN-Daten allein der **Hubraum** den Effekt trägt
  (Kraftstoff/Zylinder redundant) — und das Modell entsprechend schlank gehalten.

Diese Punkte zeigen das Arbeitsmuster im gesamten Projekt: KI liefert Tempo,
das Team liefert Idee, Prüfung, Interpretation und Entscheidung.

## Ergänzung: Quellen und wissenschaftliche Belege

Alle fachlichen und quantitativen Aussagen im Paper werden mit Quellen belegt –
u. a. die Datenquellen (Manheim/Cox Automotive, FRED der US-Notenbank, NHTSA
vPIC-API) sowie die methodische Literatur zu hedonischen Preismodellen und
Gradient-Boosting-Verfahren. KI-generierte Textentwürfe wurden nicht als Quelle
verwendet, sondern nur als Formulierungshilfe; die inhaltliche Absicherung
erfolgt ausschließlich über zitierfähige Fachquellen.
