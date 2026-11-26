import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import type { Coordinates, RouteAlternative } from "../domain/types";

type AirMapProps = {
  location: Coordinates;
  qualityColor: string;
  aqi: number | null;
  origin: Coordinates;
  destination: Coordinates;
  routes: RouteAlternative[];
  selectedRoute: number;
  pickMode: "location" | "origin" | "destination" | null;
  onPickLocation: (coordinates: Coordinates) => void;
  onSelectRoute: (index: number) => void;
};

function markerIcon(className: string): L.DivIcon {
  return L.divIcon({
    className: "",
    html: `<span class="map-marker ${className}"></span>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
}

export default function AirMap({
  location,
  qualityColor,
  aqi,
  origin,
  destination,
  routes,
  selectedRoute,
  pickMode,
  onPickLocation,
  onSelectRoute,
}: AirMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const routeLayers = useRef<L.Polyline[]>([]);
  const [map, setMap] = useState<L.Map | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const instance = L.map(containerRef.current, {
      zoomControl: false,
      attributionControl: true,
    }).setView([location.lat, location.lon], 12);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(instance);
    L.control.zoom({ position: "bottomright" }).addTo(instance);
    setMap(instance);

    return () => {
      instance.remove();
      setMap(null);
    };
  }, []);

  useEffect(() => {
    if (!map) return;
    const onMapClick = (event: L.LeafletMouseEvent) => {
      if (pickMode) {
        onPickLocation({
          lat: event.latlng.lat,
          lon: event.latlng.lng,
        });
      }
    };
    map.getContainer().style.cursor = pickMode ? "crosshair" : "";
    map.on("click", onMapClick);
    return () => {
      map.off("click", onMapClick);
      map.getContainer().style.cursor = "";
    };
  }, [map, pickMode, onPickLocation]);

  useEffect(() => {
    if (!map) return;
    const marker = L.circleMarker([location.lat, location.lon], {
      radius: 9,
      color: "#ffffff",
      weight: 3,
      fillColor: qualityColor,
      fillOpacity: 1,
    })
      .bindPopup(`Dự báo AQI: ${aqi ?? "đang cập nhật"}`)
      .addTo(map);
    map.panTo([location.lat, location.lon], { animate: true });
    return () => {
      marker.remove();
    };
  }, [map, location, qualityColor, aqi]);

  useEffect(() => {
    if (!map) return;
    const start = L.marker([origin.lat, origin.lon], {
      icon: markerIcon("map-marker--start"),
      title: "Điểm đi",
    })
      .bindTooltip("Điểm đi")
      .addTo(map);
    const end = L.marker([destination.lat, destination.lon], {
      icon: markerIcon("map-marker--end"),
      title: "Điểm đến",
    })
      .bindTooltip("Điểm đến")
      .addTo(map);
    return () => {
      start.remove();
      end.remove();
    };
  }, [map, origin, destination]);

  useEffect(() => {
    if (!map) return;
    routeLayers.current.forEach((layer) => layer.remove());
    routeLayers.current = routes.map((route, index) => {
      const selected = index === selectedRoute;
      const line = L.polyline(
        route.geometry.map(([lon, lat]) => [lat, lon]),
        {
          color: selected ? "#31785a" : "#91a89b",
          opacity: selected ? 0.95 : 0.62,
          weight: selected ? 6 : 4,
          lineCap: "round",
          lineJoin: "round",
        },
      ).addTo(map);
      line.on("click", () => onSelectRoute(index));
      line.bindTooltip(
        index === 0 ? "Tuyến ít phơi nhiễm nhất" : `Tuyến thay thế ${index}`,
      );
      return line;
    });

    if (routes.length > 0 && routeLayers.current[selectedRoute]) {
      map.fitBounds(routeLayers.current[selectedRoute].getBounds(), {
        padding: [36, 36],
        maxZoom: 14,
      });
    }

    return () => {
      routeLayers.current.forEach((layer) => layer.remove());
      routeLayers.current = [];
    };
  }, [map, routes, selectedRoute, onSelectRoute]);

  return <div className="map-canvas" ref={containerRef} aria-label="Bản đồ chất lượng không khí" />;
}
