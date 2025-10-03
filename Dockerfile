# ---------- 1. Image de base ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
ENV BASE_DIR=/data
ENV DEFAULT_PROJECT=default.qgs

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg ca-certificates curl && \
    # 🔑 SEULE clé disponible
    curl -L https://qgis.org/downloads/qgis-2021.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/qgis-archive.gpg && \
    echo "deb https://qgis.org/debian bookworm main" > /etc/apt/sources.list.d/qgis.list && \
    apt-get update

# ---------- 3. Paquets système ----------
RUN apt-get install -y --no-install-recommends \
    qgis-server \
    qgis-providers \
    python3-qgis \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    build-essential \
    # --- Dépendances Qt pour QgsApplication ---
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    libqt5xml5 \
    libqt5svg5 \
    libqt5printsupport5 \
    libqt5network5 \
    libfontconfig1 \
    libfreetype6 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ---------- 4. Répertoires ----------
RUN mkdir -p /data/projects /data/parcels /app

# ---------- 5. Dépendances Python ----------
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------- 6. Projet QGIS minimal ----------
RUN printf '%s\n' \
'<?xml version="1.0" encoding="UTF-8"?>' \
'<!DOCTYPE qgis-project>' \
'<qgis version="3.40.6-Bratislava" projectname="default">' \
'  <layer-tree-group expanded="1" checked="Qt::Checked">' \
'    <layer-tree-layer expanded="1" checked="Qt::Checked" id="parcels" name="parcels"/>' \
'  </layer-tree-group>' \
'  <projectLayers>' \
'    <layer type="vector" id="parcels" enabled="1">' \
'      <source>/data/parcels/all_parcels.geojson</source>' \
'      <provider>ogr</provider>' \
'      <crs>' \
'        <spatialrefsys>' \
'          <proj4>+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs</proj4>' \
'          <srsid>3452</srsid>' \
'          <srid>32630</srid>' \
'          <authid>EPSG:32630</authid>' \
'        </spatialrefsys>' \
'      </crs>' \
'    </layer>' \
'  </projectLayers>' \
'  <properties>' \
'    <WMSExtent>' \
'      <value>100000,1300000,400000,1600000</value>' \
'    </WMSExtent>' \
'  </properties>' \
'</qgis>' > /data/projects/default.qgs

# ---------- 7. Fichiers applicatifs ----------
COPY api.py /app/api.py
COPY init.sh /app/init.sh
RUN chmod +x /app/init.sh
WORKDIR /app

# ---------- 8. Lancement ----------
ENTRYPOINT ["/app/init.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "8", "--timeout", "0", "api:app"]