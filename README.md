# Urban Air Chitect – urban-air-chitect-ydcc

This repository contains code and data for urban air‑quality modeling.

## Directory layout
- **data/** – raw CSV files (Beijing dataset)
- **notebooks/** – exploratory Jupyter notebooks
- **src/** – source code
- **models/** – trained model artefacts

## Dataset
Beijing Multi‑Site Air‑Quality data (UCI Machine Learning Repository).  
Citation:  
Zhu, Y., & Wang, H. (2022). *Beijing Multi‑Site Air Quality Data Set*. UCI ML Repository. DOI:10.24432/C5RK5G

## Web app

The React/Vite dashboard runs the exported ONNX PM2.5 models in the browser,
shows Open-Meteo air-quality and weather data on a Leaflet map, and compares
OSRM route alternatives.

```bash
cd webapp
npm install
npm run dev
```

The build copies the ONNX models and scaler metadata from `models/` into the
static assets automatically. To create a production build, run `npm run build`;
to run the frontend tests, run `npm test`.

The dashboard requires an internet connection for Open-Meteo, OSRM, and map
tiles. Model inference runs locally in the browser. Forecasts are estimates
from a model trained on Beijing air-quality data, not official readings or
medical advice.
