import type { Coordinates, RouteAlternative } from "./types";

type OsrmRoute = {
  geometry: {
    coordinates: [number, number][];
  };
  distance: number;
  duration: number;
};

type OsrmResponse = {
  code: string;
  routes: OsrmRoute[];
};

type AirQualityResponse = {
  current?: {
    pm2_5?: number | null;
  };
};

function sampleRoute(route: OsrmRoute): Coordinates[] {
  const points = route.geometry.coordinates;
  const indexes = Array.from(
    new Set([0, Math.floor((points.length - 1) / 2), points.length - 1]),
  );
  return indexes.map((index) => ({
    lon: points[index][0],
    lat: points[index][1],
  }));
}

async function getPm25(coordinates: Coordinates): Promise<number> {
  const url = new URL("https://air-quality-api.open-meteo.com/v1/air-quality");
  url.search = new URLSearchParams({
    latitude: coordinates.lat.toFixed(4),
    longitude: coordinates.lon.toFixed(4),
    current: "pm2_5",
    timezone: "auto",
  }).toString();

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Không lấy được PM2.5 trên tuyến (${response.status}).`);
  }
  const data = (await response.json()) as AirQualityResponse;
  const pm25 = data.current?.pm2_5;
  if (pm25 === null || pm25 === undefined || !Number.isFinite(pm25)) {
    throw new Error("API không có dữ liệu PM2.5 tại một điểm trên tuyến.");
  }
  return pm25;
}

export async function findRoutes(
  origin: Coordinates,
  destination: Coordinates,
): Promise<RouteAlternative[]> {
  const url = new URL(
    `https://router.project-osrm.org/route/v1/driving/${origin.lon},${origin.lat};${destination.lon},${destination.lat}`,
  );
  url.search = new URLSearchParams({
    alternatives: "3",
    overview: "full",
    geometries: "geojson",
    steps: "false",
  }).toString();

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Không kết nối được dịch vụ tìm đường (${response.status}).`);
  }
  const data = (await response.json()) as OsrmResponse;
  if (data.code !== "Ok" || data.routes.length === 0) {
    throw new Error("Không tìm thấy tuyến đường giữa hai điểm đã chọn.");
  }

  const sampledPoints = data.routes.map(sampleRoute);
  const uniquePoints = new Map<string, Coordinates>();
  for (const points of sampledPoints) {
    for (const point of points) {
      uniquePoints.set(
        `${point.lat.toFixed(3)},${point.lon.toFixed(3)}`,
        point,
      );
    }
  }
  const measurements = await Promise.all(
    Array.from(uniquePoints.entries()).map(async ([key, point]) => [
      key,
      await getPm25(point),
    ] as const),
  );
  const pmByPoint = new Map(measurements);

  return data.routes.map((route, index) => {
    const points = sampledPoints[index];
    const readings = points.map((point) => {
      const pm25 = pmByPoint.get(
        `${point.lat.toFixed(3)},${point.lon.toFixed(3)}`,
      );
      if (pm25 === undefined) {
        throw new Error("Thiếu dữ liệu PM2.5 cho một điểm lấy mẫu trên tuyến.");
      }
      return pm25;
    });
    const averagePm25 =
      readings.reduce((sum, pm25) => sum + pm25, 0) / readings.length;
    return {
      geometry: route.geometry.coordinates,
      distance: route.distance,
      duration: route.duration,
      averagePm25,
      exposure: averagePm25 * (route.duration / 3600),
    };
  });
}
