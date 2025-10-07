# ================================================================
# Dockerfile - Image optimisée QGIS + Flask (Version Corrigée - Alignement avec GDAL système de QGIS)
# ================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    QGIS_DISABLE_MESSAGE_HOOKS=1 \
    QGIS_NO_OVERRIDE_IMPORT=1 \
    PORT=10000

# --- Dépendances système de base (sans les python3-* géospatiaux pour l'instant) ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    # Installer les bibliothèques C/C++ géospatiales système nécessaires pour QGIS
    gdal-bin \
    libgdal-dev \
    # Installer les bibliothèques de base pour python
    python3-numpy \
    python3-pandas \
    # Installer les dépendances Qt nécessaires pour QGIS
    libqt5gui5 \
    libqt5core5a \
    libqt5printsupport5 \
    libqt5svg5 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Dépôt QGIS ---
RUN wget -O - https://qgis.org/downloads/qgis-archive-keyring.gpg   | gpg --dearmor | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu   jammy main" > /etc/apt/sources.list.d/qgis.list

# --- Installer QGIS (et ses dépendances, y compris la version C/C++ de GDAL) ---
RUN apt-get update || (sleep 10 && apt-get update) \
    && apt-get install -y \
    qgis \
    qgis-server \
    qgis-plugin-grass \
    python3-qgis \
    qgis-providers \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Lire la version de GDAL C/C++ installée par le paquet QGIS ---
# Cela va déterminer les versions Python à installer via pip
RUN ogrinfo --version > /tmp/gdal_version_full.txt 2>&1 || echo "3.6.4" > /tmp/gdal_version_full.txt # Valeur par défaut si erreur
RUN cat /tmp/gdal_version_full.txt | grep -oP '\d+\.\d+\.\d+' | head -n1 > /tmp/gdal_version.txt || echo "3.6.4" > /tmp/gdal_version.txt
RUN export GDAL_VERSION=$(cat /tmp/gdal_version.txt | xargs) && \
    echo "Version GDAL C/C++ détectée (ou par défaut) : $GDAL_VERSION" && \
    # Installer les paquets Python géospatiaux *spécifiques* compatibles avec cette version GDAL
    # ATTENTION : Les versions pip doivent être compatibles avec la version GDAL C/C++
    # Consultez les notes de version de Fiona, PyProj, Shapely, GeoPandas pour la compatibilité.
    # Exemple pour GDAL 3.6.4 : Fiona 1.9.x, PyProj 3.4+, Shapely 2.0+, GeoPandas 0.14+
    # Nous installons ici des versions *connues* pour être compatibles avec GDAL 3.6.x
    # ou la dernière version stable qui l'est.
    # Pour GDAL 3.6.4, Fiona 1.9.5 + PyProj 3.6.1 + Shapely 2.0.2 + GeoPandas 0.14.1 devraient fonctionner.
    # MAIS assurez-vous que le wheel GDAL pip est compatible avec la version C/C++ installée.
    # Si les versions de requirements.txt sont incompatibles, les forcer ici.
    # On commence par installer GDAL lui-même via pip, ce qui est risqué mais parfois nécessaire.
    # Il faut trouver une version de GDAL pip qui *fonctionne* avec la version C/C++ système.
    # Essayons avec la version exacte ou la plus proche.
    # IMPORTANT : La version de GDAL pip doit être identique ou très proche de la version C/C++.
    # pip install GDAL==<version> --global-option=build_ext --global-option="-I/usr/include/gdal"
    # Cependant, cette méthode est obsolète. Utiliser des wheels binaires est préférable.
    # La meilleure façon est de trouver des wheels GDAL, Fiona, etc. compatibles.
    # Pour Ubuntu 22.04 et QGIS 3.34.x, la version système de GDAL est probablement 3.6.x.
    # On va tenter d'installer les versions de votre requirements.txt qui *devraient* être compatibles.
    # Si cela échoue, il faudra peut-être ajuster ou utiliser des wheels spécifiques.
    # Supposons que GDAL 3.6.4 C/C++ fonctionne avec GDAL pip 3.6.4.
    pip3 install --no-cache-dir --no-binary=geopandas,shapely GDAL==$GDAL_VERSION && \
    # Ensuite, installer Fiona avec une version compatible (vérifiez la compatibilité avec GDAL 3.6.4)
    # Fiona 1.9.5 est dans votre requirements.txt
    pip3 install --no-cache-dir --no-binary=:all: fiona==1.9.5 && \
    # PyProj 3.6.1 est dans votre requirements.txt
    pip3 install --no-cache-dir pyproj==3.6.1 && \
    # Shapely 2.0.2 est dans votre requirements.txt
    pip3 install --no-cache-dir shapely==2.0.2 && \
    # Enfin, installer GeoPandas 0.14.1 (celle de votre requirements.txt)
    # avec --no-deps pour éviter que pip ne réinstalle GDAL/Fiona/PyProj/Shapely
    pip3 install --no-cache-dir --no-deps geopandas==0.14.1

# --- Installation des autres dépendances Python via pip ---
# Copier requirements.txt
COPY requirements.txt /tmp/requirements.txt

# --- Filtrer requirements.txt pour enlever les bibliothèques géospatiales déjà installées ---
RUN grep -v -E '^(GDAL|fiona|pyproj|shapely|geopandas|numpy|pandas)==.*$' /tmp/requirements.txt > /tmp/requirements_filtered.txt

# Installer les paquets pip restants (sans numpy/pandas/gdal/fiona/pyproj/shapely/geopandas)
RUN pip3 install --no-cache-dir -r /tmp/requirements_filtered.txt

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
