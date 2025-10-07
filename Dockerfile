# ================================================================
# Dockerfile - Image optimisée QGIS + Flask (Stratégie de contournement)
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

# --- Vérifier les versions système installées (utile pour le debug) ---
RUN echo "--- Versions des bibliothèques géospatiales système ---" && \
    python3 -c "import gdal; print('GDAL Python (system):', gdal.__version__)" || echo "GDAL Python (system) not found" && \
    python3 -c "import fiona; print('Fiona Python (system):', fiona.__version__)" || echo "Fiona Python (system) not found" && \
    python3 -c "import pyproj; print('PyProj Python (system):', pyproj.__version__)" || echo "PyProj Python (system) not found" && \
    python3 -c "import shapely; print('Shapely Python (system):', shapely.__version__)" || echo "Shapely Python (system) not found" && \
    python3 -c "import geopandas; print('GeoPandas Python (system):', geopandas.__version__)" || echo "GeoPandas Python (system) not found" && \
    ogrinfo --version || echo "GDAL CLI not found"

# --- Installation dépendances Python via pip ---
# Copier requirements.txt
COPY requirements.txt /tmp/requirements.txt

# --- Filtrer requirements.txt pour enlever les bibliothèques géospatiales ---
# et installer les autres paquets avec --no-deps
RUN grep -v -E '^(GDAL|fiona|pyproj|shapely|geopandas)==.*$' /tmp/requirements.txt > /tmp/requirements_filtered.txt

# Installer les paquets pip restants sans réinstaller les dépendances (y compris numpy/pandas)
RUN pip3 install --no-cache-dir --no-deps -r /tmp/requirements_filtered.txt

# Installer explicitement geopandas et shapely qui dépendent de numpy/pandas/fiona/pyproj/gdal
# mais en s'assurant qu'ils utilisent les versions système (si possible via apt ou en les installant séparément sans version spécifique)
# Si pip les réinstalle, cela peut annuler la stratégie. On les réinstalle donc *sans version spécifique* pour qu'ils utilisent les dépendances système.
# ATTENTION: Cela peut réinstaller Fiona/GDAL via pip si geopandas/shapely les exigent explicitement comme dépendances.
# Une alternative est de ne *PAS* les installer ici, et de s'assurer que python3-geopandas installé via apt suffit.
# Cependant, la version dans apt peut ne pas correspondre à 0.14.1.
# Essayons d'abord sans les réinstaller via pip.
# RUN pip3 install --no-cache-dir geopandas shapely

# Option 2 : Si la version de geopandas/shapely installée par apt est trop ancienne, et que vous devez utiliser pip,
# forcez l'installation *sans réinstaller les dépendances*.
# RUN pip3 install --no-cache-dir --no-deps geopandas==0.14.1 shapely==2.0.2

# Option 1 (préférée pour commencer) : Ne pas réinstaller geopandas/shapely via pip, supposer que python3-geopandas est suffisant.
# Option 2 (si nécessaire) : Décommenter la ligne ci-dessous. Cela réinstallera *potentiellement* Fiona/GDAL via pip.
# RUN pip3 install --no-cache-dir --no-deps geopandas==0.14.1 shapely==2.0.2 pyproj==3.6.1

# Si vous choisissez l'Option 2, assurez-vous que les versions pip sont *vraiment* compatibles avec la version GDAL système.
# Vous pouvez les lire comme dans le Dockerfile précédent et les forcer ici.
RUN python3 -c "import gdal; print(gdal.__version__)" > /tmp/gdal_version.txt 2>/dev/null || echo "3.4.3" > /tmp/gdal_version.txt
RUN python3 -c "import fiona; print(fiona.__version__)" > /tmp/fiona_version.txt 2>/dev/null || echo "1.8.22" > /tmp/fiona_version.txt
RUN python3 -c "import pyproj; print(pyproj.__version__)" > /tmp/pyproj_version.txt 2>/dev/null || echo "3.3.0" > /tmp/pyproj_version.txt
RUN python3 -c "import shapely; print(shapely.__version__)" > /tmp/shapely_version.txt 2>/dev/null || echo "1.8.5" > /tmp/shapely_version.txt
RUN python3 -c "import geopandas; print(geopandas.__version__)" > /tmp/geopandas_version.txt 2>/dev/null || echo "0.12.2" > /tmp/geopandas_version.txt

RUN export GDAL_VERSION=$(cat /tmp/gdal_version.txt | head -n1 | xargs) && \
    export FIONA_VERSION=$(cat /tmp/fiona_version.txt | head -n1 | xargs) && \
    export PYPROJ_VERSION=$(cat /tmp/pyproj_version.txt | head -n1 | xargs) && \
    export SHAPELY_VERSION=$(cat /tmp/shapely_version.txt | head -n1 | xargs) && \
    export GEOPANDAS_VERSION=$(cat /tmp/geopandas_version.txt | head -n1 | xargs) && \
    echo "Versions système lues : GDAL=$GDAL_VERSION, Fiona=$FIONA_VERSION, PyProj=$PYPROJ_VERSION, Shapely=$SHAPELY_VERSION, GeoPandas=$GEOPANDAS_VERSION" && \
    # Installer les versions spécifiques *compatibles* via pip, sans réinstaller leurs dépendances
    pip3 install --no-cache-dir --no-deps GDAL==$GDAL_VERSION && \
    pip3 install --no-cache-dir --no-deps fiona==$FIONA_VERSION && \
    pip3 install --no-cache-dir --no-deps pyproj==$PYPROJ_VERSION && \
    pip3 install --no-cache-dir --no-deps shapely==$SHAPELY_VERSION && \
    pip3 install --no-cache-dir --no-deps geopandas==$GEOPANDAS_VERSION

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
