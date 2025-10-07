# ================================================================
# Dockerfile - Image optimisée QGIS + Flask (Version Corrigée - Alignement avec apt)
# ================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    QGIS_DISABLE_MESSAGE_HOOKS=1 \
    QGIS_NO_OVERRIDE_IMPORT=1 \
    PORT=10000

# --- Dépendances système de base ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    # Installer les bibliothèques géospatiales système AVANT QGIS
    gdal-bin \
    libgdal-dev \
    # Paquets Python correspondants aux bibliothèques géospatiales système
    python3-gdal \
    python3-fiona \
    python3-shapely \
    python3-pyproj \
    python3-numpy \
    python3-pandas \
    # Paquets QGIS
    libqt5gui5 \
    libqt5core5a \
    libqt5printsupport5 \
    libqt5svg5 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Dépôt QGIS ---
RUN wget -O - https://qgis.org/downloads/qgis-archive-keyring.gpg   | gpg --dearmor | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu   jammy main" > /etc/apt/sources.list.d/qgis.list

# --- Installer QGIS (et ses dépendances) ---
RUN apt-get update || (sleep 10 && apt-get update) \
    && apt-get install -y \
    qgis \
    qgis-server \
    qgis-plugin-grass \
    python3-qgis \
    qgis-providers \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Lire les versions installées via apt (nécessite python3-gdal/fiona/pyproj installés) ---
RUN python3 -c "import gdal; print(gdal.__version__)" > /tmp/gdal_version.txt 2>/dev/null || echo "3.4.3" > /tmp/gdal_version.txt # Valeur par défaut si erreur
RUN python3 -c "import fiona; print(fiona.__version__)" > /tmp/fiona_version.txt 2>/dev/null || echo "1.8.22" > /tmp/fiona_version.txt # Valeur par défaut si erreur
RUN python3 -c "import pyproj; print(pyproj.__version__)" > /tmp/pyproj_version.txt 2>/dev/null || echo "3.3.0" > /tmp/pyproj_version.txt # Valeur par défaut si erreur
RUN python3 -c "import shapely; print(shapely.__version__)" > /tmp/shapely_version.txt 2>/dev/null || echo "1.8.5" > /tmp/shapely_version.txt # Valeur par défaut si erreur
RUN python3 -c "import geopandas; print(geopandas.__version__)" > /tmp/geopandas_version.txt 2>/dev/null || echo "0.12.2" > /tmp/geopandas_version.txt # Valeur par défaut si erreur

# --- Installation dépendances Python via pip ---
# Copier requirements.txt
COPY requirements.txt /tmp/requirements.txt

# --- Créer une version modifiée de requirements.txt avec les versions système ---
# On va remplacer les lignes GDAL, fiona, pyproj, shapely, geopandas par les versions lues
# ou des versions compatibles connues pour fonctionner avec QGIS 3.34.x sur Ubuntu 22.04
# (Valeurs typiques pour qgis.org/ubuntu jammy, à ajuster si nécessaire)

# Lire les versions
RUN export GDAL_VERSION=$(cat /tmp/gdal_version.txt | head -n1 | xargs) && \
    export FIONA_VERSION=$(cat /tmp/fiona_version.txt | head -n1 | xargs) && \
    export PYPROJ_VERSION=$(cat /tmp/pyproj_version.txt | head -n1 | xargs) && \
    export SHAPELY_VERSION=$(cat /tmp/shapely_version.txt | head -n1 | xargs) && \
    export GEOPANDAS_VERSION=$(cat /tmp/geopandas_version.txt | head -n1 | xargs) && \
    echo "Versions détectées (ou par défaut) : GDAL=$GDAL_VERSION, Fiona=$FIONA_VERSION, PyProj=$PYPROJ_VERSION, Shapely=$SHAPELY_VERSION, GeoPandas=$GEOPANDAS_VERSION" && \
    # Générer une liste de paquets à installer avec pip, en forçant les versions géospatiales
    # et en excluant celles du fichier original qui pourraient entrer en conflit
    echo "flask==3.0.0" > /tmp/requirements_final.txt && \
    echo "flask-cors==4.0.0" >> /tmp/requirements_final.txt && \
    echo "flask-compress==1.14" >> /tmp/requirements_final.txt && \
    echo "werkzeug==3.0.1" >> /tmp/requirements_final.txt && \
    echo "pydantic==2.5.0" >> /tmp/requirements_final.txt && \
    # Installer les versions spécifiques de GDAL, Fiona, PyProj, Shapely, GeoPandas
    echo "GDAL==$GDAL_VERSION" >> /tmp/requirements_final.txt && \
    echo "fiona==$FIONA_VERSION" >> /tmp/requirements_final.txt && \
    echo "pyproj==$PYPROJ_VERSION" >> /tmp/requirements_final.txt && \
    echo "shapely==$SHAPELY_VERSION" >> /tmp/requirements_final.txt && \
    echo "geopandas==$GEOPANDAS_VERSION" >> /tmp/requirements_final.txt && \
    # Ajouter les autres paquets de l'original, sauf ceux potentiellement en conflit
    grep -v -E '^(GDAL|fiona|pyproj|shapely|geopandas|numpy|pandas)==' /tmp/requirements.txt | grep -v -E '^(numpy|pandas)==$' >> /tmp/requirements_final.txt

# Installer les paquets pip avec les versions corrigées
RUN pip3 install --no-cache-dir -r /tmp/requirements_final.txt

# Création structure dossiers
RUN mkdir -p /opt/render/project/src/data/{shapefiles,csv,geojson,projects,other,tiles,parcels,documents,cache}

# Copie application
WORKDIR /opt/render/project/src
COPY api.py .
COPY default.qgs data/projects/default.qgs

# Permissions
RUN chmod -R 755 /opt/render/project/src && \
    chmod -R 777 /opt/render/project/src/data

# Exposition port
EXPOSE 10000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:10000/api/health || exit 1

# Démarrage avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "1", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api:app"]
