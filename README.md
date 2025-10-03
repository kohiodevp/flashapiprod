# API QGIS Server - Optimisée pour Applications Mobiles Géolocalisées

## 📋 Vue d'ensemble

API REST haute performance combinant QGIS Server et Flask, spécialement optimisée pour applications mobiles géolocalisées (Ionic/Angular/Leaflet). Déploiement sur Render avec support complet des projections géographiques (EPSG:32630 par défaut).

### Fonctionnalités principales

- **Services OGC** : WMS/WFS/WCS compatibles Leaflet
- **Gestion parcelles cadastrales** : Création, consultation, analyse
- **Transformations géographiques** : Conversion entre systèmes de coordonnées
- **Rapports PDF** : Génération automatique avec mise en cache
- **Requêtes spatiales optimisées** : Récupération par emprise visible (viewport mobile)
- **Sessions persistantes** : Gestion multi-utilisateurs
- **Cache intelligent** : Redis ou fallback fichier

## 🚀 Déploiement rapide sur Render

### Prérequis

- Compte GitHub
- Compte Render (gratuit)
- Git installé

### Étapes

```bash
# 1. Cloner et configurer
git clone https://github.com/votre-compte/qgis-api.git
cd qgis-api

# 2. Vérifier fichiers
ls -la
# Doit contenir: api.py, Dockerfile, render.yaml, requirements.txt, default.qgs

# 3. Push vers GitHub
git remote set-url origin https://github.com/votre-compte/qgis-api.git
git push -u origin main

# 4. Sur Render Dashboard
# - New > Blueprint
# - Connecter repository
# - Deploy automatique démarre
```

### Variables d'environnement

```env
PORT=10000
DEFAULT_CRS=EPSG:32630
BASE_DIR=/opt/render/project/src/data
DEBUG=false
REDIS_URL=redis://... (optionnel)
```

## 📱 Intégration Ionic/Angular

### Installation dépendances

```bash
npm install leaflet @types/leaflet
npm install @capacitor/geolocation
npm install @capacitor/filesystem
```

### Configuration Ionic (capacitor.config.ts)

```typescript
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.votre.app',
  appName: 'QGIS Mobile',
  webDir: 'www',
  server: {
    androidScheme: 'https'
  },
  plugins: {
    Geolocation: {
      permissions: {
        location: 'always'
      }
    }
  }
};

export default config;
```

### Service API (qgis-api.service.ts)

```typescript
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

interface Parcel {
  id: string;
  name: string;
  commune: string;
  section: string;
  numero: string;
  superficie_ha: number;
  geometry: any;
}

interface BoundsQuery {
  minx: number;
  miny: number;
  maxx: number;
  maxy: number;
  crs: string;
  buffer_m?: number;
}

@Injectable({
  providedIn: 'root'
})
export class QgisApiService {
  private apiUrl = 'https://votre-app.onrender.com/api';
  private sessionId: string | null = null;

  constructor(private http: HttpClient) {
    this.sessionId = localStorage.getItem('qgis_session_id');
  }

  private getHeaders(): HttpHeaders {
    let headers = new HttpHeaders({'Content-Type': 'application/json'});
    if (this.sessionId) {
      headers = headers.set('X-Session-ID', this.sessionId);
    }
    return headers;
  }

  createParcel(parcelData: any): Observable<Parcel> {
    return this.http.post<Parcel>(`${this.apiUrl}/parcels`, parcelData, {
      headers: this.getHeaders(),
      observe: 'response'
    }).pipe(
      tap(response => {
        const sessionId = response.headers.get('X-Session-ID');
        if (sessionId) {
          this.sessionId = sessionId;
          localStorage.setItem('qgis_session_id', sessionId);
        }
      }),
      map(response => response.body)
    );
  }

  getParcelsByBounds(bounds: BoundsQuery): Observable<any> {
    return this.http.post(`${this.apiUrl}/parcels/bounds`, bounds, {
      headers: this.getHeaders()
    });
  }

  getParcel(parcelId: string, crs = 'EPSG:4326'): Observable<Parcel> {
    return this.http.get<Parcel>(`${this.apiUrl}/parcels/${parcelId}?crs=${crs}`, {
      headers: this.getHeaders()
    });
  }

  deleteParcel(parcelId: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/parcels/${parcelId}`, {
      headers: this.getHeaders()
    });
  }

  transformCoordinates(coords: number[][], fromCrs: string, toCrs: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/crs/transform`, {
      coordinates: coords,
      from_crs: fromCrs,
      to_crs: toCrs
    }, { headers: this.getHeaders() });
  }

  downloadReport(parcelId: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/reports/parcel/${parcelId}`, {
      headers: this.getHeaders(),
      responseType: 'blob'
    });
  }

  getWmsUrl(): string {
    return `${this.apiUrl}/ogc/WMS`;
  }
}
```

### Composant carte (map.page.ts)

```typescript
import { Component, OnInit, OnDestroy } from '@angular/core';
import * as L from 'leaflet';
import { Geolocation } from '@capacitor/geolocation';
import { QgisApiService } from '../services/qgis-api.service';
import { ToastController, LoadingController, AlertController } from '@ionic/angular';

@Component({
  selector: 'app-map',
  templateUrl: './map.page.html',
  styleUrls: ['./map.page.scss']
})
export class MapPage implements OnInit, OnDestroy {
  private map: L.Map;
  private parcelsLayer: L.GeoJSON;
  private wmsLayer: L.TileLayer.WMS;
  private userMarker: L.Marker;
  private drawnItems: L.FeatureGroup;

  constructor(
    private qgisApi: QgisApiService,
    private toastCtrl: ToastController,
    private loadingCtrl: LoadingController,
    private alertCtrl: AlertController
  ) {}

  async ngOnInit() {
    await this.initMap();
    await this.loadUserLocation();
  }

  private async initMap(): Promise<void> {
    // Initialisation carte
    this.map = L.map('map', {
      center: [12.3714, -1.5197], // Ouagadougou
      zoom: 13,
      zoomControl: true
    });

    // Fond de carte
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap'
    }).addTo(this.map);

    // Couche WMS QGIS
    this.wmsLayer = L.tileLayer.wms(this.qgisApi.getWmsUrl(), {
      layers: 'Parcelles',
      format: 'image/png',
      transparent: true,
      version: '1.3.0',
      crs: L.CRS.EPSG4326
    }).addTo(this.map);

    // Layer pour parcelles GeoJSON
    this.parcelsLayer = L.geoJSON(null, {
      style: {
        color: '#3388ff',
        weight: 2,
        fillOpacity: 0.2
      },
      onEachFeature: (feature, layer) => {
        layer.on('click', () => this.onParcelClick(feature));
      }
    }).addTo(this.map);

    // Layer pour dessins
    this.drawnItems = new L.FeatureGroup();
    this.map.addLayer(this.drawnItems);

    // Contrôles de dessin
    const drawControl = new L.Control.Draw({
      edit: {
        featureGroup: this.drawnItems
      },
      draw: {
        polygon: true,
        polyline: false,
        circle: false,
        rectangle: true,
        marker: false,
        circlemarker: false
      }
    });
    this.map.addControl(drawControl);

    // Événements
    this.map.on(L.Draw.Event.CREATED, (e: any) => {
      this.drawnItems.addLayer(e.layer);
      this.onDrawComplete(e.layer);
    });

    this.map.on('moveend', () => this.loadParcelsInView());
    
    // Chargement initial
    await this.loadParcelsInView();
  }

  private async loadUserLocation(): Promise<void> {
    try {
      const loading = await this.loadingCtrl.create({
        message: 'Localisation...'
      });
      await loading.present();

      const position = await Geolocation.getCurrentPosition({
        enableHighAccuracy: true,
        timeout: 10000
      });

      const coords: L.LatLngExpression = [
        position.coords.latitude,
        position.coords.longitude
      ];

      // Marqueur utilisateur
      const userIcon = L.icon({
        iconUrl: 'assets/icons/user-location.png',
        iconSize: [32, 32],
        iconAnchor: [16, 32]
      });

      if (this.userMarker) {
        this.userMarker.setLatLng(coords);
      } else {
        this.userMarker = L.marker(coords, { icon: userIcon })
          .bindPopup('Vous êtes ici')
          .addTo(this.map);
      }

      this.map.setView(coords, 15);
      await loading.dismiss();

    } catch (error) {
      console.error('Erreur géolocalisation:', error);
      const toast = await this.toastCtrl.create({
        message: 'Impossible de vous localiser',
        duration: 3000,
        color: 'warning'
      });
      await toast.present();
    }
  }

  private async loadParcelsInView(): Promise<void> {
    const bounds = this.map.getBounds();
    
    const boundsQuery = {
      minx: bounds.getWest(),
      miny: bounds.getSouth(),
      maxx: bounds.getEast(),
      maxy: bounds.getNorth(),
      crs: 'EPSG:4326',
      buffer_m: 50
    };

    this.qgisApi.getParcelsByBounds(boundsQuery).subscribe({
      next: (response) => {
        this.parcelsLayer.clearLayers();
        if (response.features && response.features.length > 0) {
          this.parcelsLayer.addData({
            type: 'FeatureCollection',
            features: response.features
          });
        }
      },
      error: async (error) => {
        console.error('Erreur chargement parcelles:', error);
      }
    });
  }

  private async onParcelClick(feature: any): Promise<void> {
    const alert = await this.alertCtrl.create({
      header: feature.properties.name,
      message: `
        <strong>Commune:</strong> ${feature.properties.commune}<br>
        <strong>Section:</strong> ${feature.properties.section}<br>
        <strong>Numéro:</strong> ${feature.properties.numero}<br>
        <strong>Superficie:</strong> ${feature.properties.superficie_ha} ha
      `,
      buttons: [
        {
          text: 'Fermer',
          role: 'cancel'
        },
        {
          text: 'Rapport PDF',
          handler: () => this.downloadReport(feature.properties.id)
        },
        {
          text: 'Supprimer',
          role: 'destructive',
          handler: () => this.deleteParcel(feature.properties.id)
        }
      ]
    });
    await alert.present();
  }

  private async onDrawComplete(layer: L.Layer): Promise<void> {
    const geometry = (layer as any).toGeoJSON().geometry;
    
    const alert = await this.alertCtrl.create({
      header: 'Nouvelle parcelle',
      inputs: [
        { name: 'name', type: 'text', placeholder: 'Nom' },
        { name: 'commune', type: 'text', placeholder: 'Commune' },
        { name: 'section', type: 'text', placeholder: 'Section' },
        { name: 'numero', type: 'text', placeholder: 'Numéro' }
      ],
      buttons: [
        { text: 'Annuler', role: 'cancel' },
        {
          text: 'Créer',
          handler: async (data) => {
            const loading = await this.loadingCtrl.create({
              message: 'Création...'
            });
            await loading.present();

            const parcelData = {
              ...data,
              geometry: geometry,
              crs: 'EPSG:4326'
            };

            this.qgisApi.createParcel(parcelData).subscribe({
              next: async (response) => {
                await loading.dismiss();
                const toast = await this.toastCtrl.create({
                  message: 'Parcelle créée avec succès',
                  duration: 2000,
                  color: 'success'
                });
                await toast.present();
                this.drawnItems.clearLayers();
                this.loadParcelsInView();
              },
              error: async (error) => {
                await loading.dismiss();
                const toast = await this.toastCtrl.create({
                  message: 'Erreur lors de la création',
                  duration: 3000,
                  color: 'danger'
                });
                await toast.present();
              }
            });
          }
        }
      ]
    });
    await alert.present();
  }

  private async downloadReport(parcelId: string): Promise<void> {
    const loading = await this.loadingCtrl.create({
      message: 'Génération PDF...'
    });
    await loading.present();

    this.qgisApi.downloadReport(parcelId).subscribe({
      next: async (blob) => {
        await loading.dismiss();
        
        // Sauvegarde fichier
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `parcelle_${parcelId}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        const toast = await this.toastCtrl.create({
          message: 'Rapport téléchargé',
          duration: 2000,
          color: 'success'
        });
        await toast.present();
      },
      error: async (error) => {
        await loading.dismiss();
        const toast = await this.toastCtrl.create({
          message: 'Erreur génération rapport',
          duration: 3000,
          color: 'danger'
        });
        await toast.present();
      }
    });
  }

  private async deleteParcel(parcelId: string): Promise<void> {
    this.qgisApi.deleteParcel(parcelId).subscribe({
      next