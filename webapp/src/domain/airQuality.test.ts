import { describe, expect, it } from "vitest";
import {
  buildInputVector,
  type AirResponse,
  type ModelConfig,
  type WeatherResponse,
} from "./airQuality";
import { calculateAqi, getAirQualityLevel } from "./aqi";

const featureColumns = [
  "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN",
  "WSPM", "PM2.5_lag_1h", "PM2.5_lag_2h", "PM2.5_lag_3h", "PM2.5_lag_6h",
  "PM2.5_lag_12h", "PM2.5_lag_24h", "PM2.5_roll_mean_6h",
  "PM2.5_roll_std_6h", "PM2.5_roll_mean_12h", "PM2.5_roll_std_12h",
  "PM2.5_roll_mean_24h", "PM2.5_roll_std_24h", "hour_sin", "hour_cos",
  "month_sin", "month_cos", "day_of_week", "wind_u", "wind_v",
];

const timestamps = Array.from({ length: 30 }, (_, index) =>
  new Date(Date.UTC(2026, 0, 5, index)).toISOString().slice(0, 16),
);

function makeWeather(): WeatherResponse {
  const constant = (value: number) => timestamps.map(() => value);
  return {
    hourly: {
      time: timestamps,
      temperature_2m: constant(12),
      dew_point_2m: constant(3),
      pressure_msl: constant(1012),
      rain: constant(0),
      wind_speed_10m: constant(4),
      wind_direction_10m: constant(90),
      relative_humidity_2m: constant(55),
    },
    utc_offset_seconds: 0,
  };
}

function makeAir(): AirResponse {
  const constant = (value: number) => timestamps.map(() => value);
  return {
    hourly: {
      time: timestamps,
      pm2_5: timestamps.map((_, index) => index),
      pm10: constant(20),
      sulphur_dioxide: constant(2),
      nitrogen_dioxide: constant(10),
      carbon_monoxide: constant(300),
      ozone: constant(30),
    },
  };
}

const config: ModelConfig = {
  feature_columns: featureColumns,
  mean: featureColumns.map(() => 0),
  scale: featureColumns.map(() => 1),
};

describe("buildInputVector", () => {
  it("preserves training feature order and derives lag, rolling and time features", () => {
    const vector = buildInputVector(makeWeather(), makeAir(), 25, config);
    const at = (column: string) => vector[featureColumns.indexOf(column)];

    expect(vector).toHaveLength(29);
    expect(at("PM2.5_lag_1h")).toBe(24);
    expect(at("PM2.5_lag_24h")).toBe(1);
    expect(at("PM2.5_roll_mean_6h")).toBe(21.5);
    expect(at("PM2.5_roll_std_6h")).toBeCloseTo(Math.sqrt(3.5), 5);
    expect(at("hour_sin")).toBeCloseTo(Math.sin(Math.PI / 12), 6);
    expect(at("month_cos")).toBe(1);
    expect(at("day_of_week")).toBe(1);
    expect(at("wind_u")).toBeCloseTo(-4, 6);
  });

  it("rejects incomplete historical PM2.5 instead of imputing a lag", () => {
    const air = makeAir();
    air.hourly.pm2_5[23] = null;

    expect(() => buildInputVector(makeWeather(), air, 25, config)).toThrow(
      "PM2.5 lịch sử",
    );
  });
});

describe("AQI", () => {
  it("maps PM2.5 breakpoints to the expected AQI range", () => {
    expect(calculateAqi(12)).toBe(50);
    expect(calculateAqi(12.1)).toBe(51);
    expect(calculateAqi(35.5)).toBe(101);
    expect(calculateAqi(-2)).toBe(0);
  });

  it("provides an appropriate health category", () => {
    expect(getAirQualityLevel(50).label).toBe("Tốt");
    expect(getAirQualityLevel(151).label).toBe("Không tốt");
  });
});
