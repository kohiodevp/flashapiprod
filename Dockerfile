# ---------- 1. Image de base ----------
FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=10000
ENV BASE_DIR=/data
ENV DEFAULT_PROJECT=default.qgs

# --- Dépendances système ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    libgdal-dev \
    libqt5gui5 \
    libqt5core5a \
    libqt5printsupport5 \
    libqt5svg5 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Dépôt QGIS ---
RUN wget -O - https://qgis.org/downloads/qgis-archive-keyring.gpg | gpg --dearmor | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" > /etc/apt/sources.list.d/qgis.list

# --- Installer QGIS ---
RUN apt-get update || (sleep 10 && apt-get update) \
    && apt-get install -y \
    qgis \
    qgis-server \
    qgis-plugin-grass \
    python3-qgis \
    qgis-providers \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Variables d'environnement QGIS ---
ENV QGIS_PREFIX_PATH="/usr"
ENV PYTHONPATH=/usr/share/qgis/python
ENV QT_QPA_PLATFORM=offscreen
ENV QT_DEBUG_PLUGINS=0

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