export type Coordinates = {
  lat: number;
  lon: number;
};

export type RouteAlternative = {
  geometry: [number, number][];
  distance: number;
  duration: number;
  averagePm25: number;
  exposure: number;
};
