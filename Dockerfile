# ---------- 1. Image de base légère ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
ENV QGIS_PROJECT_FILE=/etc/qgis/projects/project.qgs
ENV DEFAULT_CRS=EPSG:32630

# ---------- 2. Paquets système ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    wget \
    ca-certificates \
    qgis-server \
    qgis-providers \
    python3-qgis \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- 3. Répertoires ----------
RUN mkdir -p /etc/qgis/projects /app /data

# ---------- 4. Dépendances Python ----------
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------- 5. Projet QGIS minimal ----------
RUN printf '%s\n' \
'<?xml version="1.0" encoding="UTF-8"?>' \
'<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd " "SYSTEM">' \
'<qgis projectname="Render UTM30N" version="3.34.0">' \
'  <title>Render QGIS Server - EPSG:32630</title>' \
'  <spatialrefsys>' \
'    <authid>EPSG:32630</authid>' \
'    <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>' \
'    <description>WGS 84 / UTM zone 30N</description>' \
'  </spatialrefsys>' \
'  <layer-tree-group><customproperties/></layer-tree-group>' \
'</qgis>' > $QGIS_PROJECT_FILE

# ---------- 6. Code applicatif ----------
COPY api.py /app/api.py
WORKDIR /app

# ---------- 7. Start simple ----------
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api:app