# Equipos Fuchi (Streamlit)

App en Streamlit para **generar equipos equilibrados 8 vs 8** desde un Excel de jugadores, con:
- Persistencia segura a **CSV/JSON** (no sobrescribe el Excel)
- Edición de jugadores desde la UI
- Recomendaciones de cambios entre equipos
- Historial de cambios manuales + “aprendizaje” heurístico

## Requisitos
- Python 3.11+

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Ejecutar

```bash
python -m streamlit run app.py
```

## Persistencia (carpeta `data/`)
- `data/players.csv`: base de jugadores editable
- `data/config.json`: ponderaciones (físico/fútbol) y parámetros
- `data/history.json`: historial de movimientos manuales
- `data/learning.json`: agregados heurísticos a partir del historial

