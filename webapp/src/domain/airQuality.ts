import type { InferenceSession } from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import type { Coordinates } from "./types";

const weatherEndpoint = "https://api.open-meteo.com/v1/forecast";
const airEndpoint = "https://air-quality-api.open-meteo.com/v1/air-quality";
const lagHours = [1, 2, 3, 6, 12, 24] as const;
const rollingHours = [6, 12, 24] as const;

export type ModelConfig = {
  feature_columns: string[];
  mean: number[];
  scale: number[];
};

export type WeatherResponse = {
  hourly: {
    time: string[];
    temperature_2m: (number | null)[];
    dew_point_2m: (number | null)[];
    pressure_msl: (number | null)[];
    rain: (number | null)[];
    wind_speed_10m: (number | null)[];
    wind_direction_10m: (number | null)[];
    relative_humidity_2m: (number | null)[];
  };
  utc_offset_seconds: number;
};

export type AirResponse = {
  hourly: {
    time: string[];
    pm2_5: (number | null)[];
    pm10: (number | null)[];
    sulphur_dioxide: (number | null)[];
    nitrogen_dioxide: (number | null)[];
    carbon_monoxide: (number | null)[];
    ozone: (number | null)[];
  };
};

export type AirSnapshot = {
  coordinates: Coordinates;
  pm25: number;
  lower: number;
  upper: number;
  aqi: number;
  temperature: number;
  windSpeed: number;
  humidity: number;
  rain: number;
  pressure: number;
  measuredAt: string;
  history: { time: string; pm25: number }[];
};

const modelNames = {
  point: "xgboost_pm25_point.onnx",
  lower: "xgboost_pm25_q05.onnx",
  upper: "xgboost_pm25_q95.onnx",
} as const;

const modelSessions = new Map<string, Promise<InferenceSession>>();
let modelConfigPromise: Promise<ModelConfig> | undefined;
let runtimePromise: Promise<typeof import("onnxruntime-web/wasm")> | undefined;
const snapshotCache = new Map<
  string,
  { expiresAt: number; snapshot: AirSnapshot }
>();
const snapshotRequests = new Map<string, Promise<AirSnapshot>>();
const snapshotCacheMs = 5 * 60 * 1000;

function modelAssetUrl(name: string): string {
  return new URL(`${import.meta.env.BASE_URL}models/${name}`, window.location.href).href;
}

function getModelConfig(): Promise<ModelConfig> {
  if (!modelConfigPromise) {
    modelConfigPromise = fetch(modelAssetUrl("airwise_scaler.json"))
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Không tải được cấu hình mô hình (${response.status}).`);
        }
        return (await response.json()) as ModelConfig;
      })
      .then(validateModelConfig)
      .catch((error: unknown) => {
        modelConfigPromise = undefined;
        throw error;
      });
  }
  return modelConfigPromise;
}

function getSession(name: string): Promise<InferenceSession> {
  let session = modelSessions.get(name);
  if (!session) {
    session = getRuntime()
      .then(({ InferenceSession: createSession }) =>
        createSession.create(modelAssetUrl(name), {
          executionProviders: ["wasm"],
        }),
      )
      .catch((error: unknown) => {
        modelSessions.delete(name);
        throw new Error(`Không tải được mô hình ${name}.`, { cause: error });
      });
    modelSessions.set(name, session);
  }
  return session;
}

function getRuntime(): Promise<typeof import("onnxruntime-web/wasm")> {
  if (!runtimePromise) {
    runtimePromise = import("onnxruntime-web/wasm")
      .then((runtime) => {
        runtime.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
        return runtime;
      })
      .catch((error: unknown) => {
        runtimePromise = undefined;
        throw error;
      });
  }
  return runtimePromise;
}

function getValue(value: number | null | undefined, field: string): number {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    throw new Error(`Thiếu dữ liệu ${field} tại thời điểm dự báo.`);
  }
  return value;
}

function mean(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function sampleStandardDeviation(values: number[]): number {
  const average = mean(values);
  const squaredDifference = values.reduce(
    (total, value) => total + (value - average) ** 2,
    0,
  );
  return Math.sqrt(squaredDifference / (values.length - 1));
}

function getLocalCalendar(time: string): {
  hour: number;
  month: number;
  dayOfWeek: number;
} {
  const [date, clock] = time.split("T");
  const [year, month, day] = date.split("-").map(Number);
  const [hour] = clock.split(":").map(Number);
  const sundayFirstDay = new Date(Date.UTC(year, month - 1, day)).getUTCDay();
  return {
    hour,
    month,
    dayOfWeek: (sundayFirstDay + 6) % 7,
  };
}

export function buildInputVector(
  weather: WeatherResponse,
  air: AirResponse,
  timeIndex: number,
  config: ModelConfig,
): Float32Array {
  const time = weather.hourly.time[timeIndex];
  const airIndex = air.hourly.time.indexOf(time);
  if (airIndex < 24) {
    throw new Error("Không đủ dữ liệu PM2.5 trong 24 giờ trước để dự báo.");
  }

  const priorPm25 = (offset: number) =>
    getValue(air.hourly.pm2_5[airIndex - offset], "PM2.5 lịch sử");
  const values: Record<string, number> = {
    PM10: getValue(air.hourly.pm10[airIndex], "PM10"),
    SO2: getValue(air.hourly.sulphur_dioxide[airIndex], "SO2"),
    NO2: getValue(air.hourly.nitrogen_dioxide[airIndex], "NO2"),
    CO: getValue(air.hourly.carbon_monoxide[airIndex], "CO"),
    O3: getValue(air.hourly.ozone[airIndex], "O3"),
    TEMP: getValue(weather.hourly.temperature_2m[timeIndex], "nhiệt độ"),
    PRES: getValue(weather.hourly.pressure_msl[timeIndex], "áp suất"),
    DEWP: getValue(weather.hourly.dew_point_2m[timeIndex], "điểm sương"),
    RAIN: getValue(weather.hourly.rain[timeIndex], "lượng mưa"),
    WSPM: getValue(weather.hourly.wind_speed_10m[timeIndex], "tốc độ gió"),
  };

  for (const lag of lagHours) {
    values[`PM2.5_lag_${lag}h`] = priorPm25(lag);
  }
  for (const window of rollingHours) {
    const observations = Array.from({ length: window }, (_, index) =>
      priorPm25(index + 1),
    );
    values[`PM2.5_roll_mean_${window}h`] = mean(observations);
    values[`PM2.5_roll_std_${window}h`] =
      sampleStandardDeviation(observations);
  }

  const { hour, month, dayOfWeek } = getLocalCalendar(time);
  values.hour_sin = Math.sin((2 * Math.PI * hour) / 24);
  values.hour_cos = Math.cos((2 * Math.PI * hour) / 24);
  values.month_sin = Math.sin((2 * Math.PI * (month - 1)) / 12);
  values.month_cos = Math.cos((2 * Math.PI * (month - 1)) / 12);
  values.day_of_week = dayOfWeek;

  const direction = getValue(
    weather.hourly.wind_direction_10m[timeIndex],
    "hướng gió",
  );
  const radians = (direction * Math.PI) / 180;
  const speed = values.WSPM;
  values.wind_u = -speed * Math.sin(radians);
  values.wind_v = -speed * Math.cos(radians);

  return Float32Array.from(
    config.feature_columns.map((column, index) => {
      const value = values[column];
      if (!Number.isFinite(value)) {
        throw new Error(`Không tạo được đặc trưng ${column}.`);
      }
      return (value - config.mean[index]) / config.scale[index];
    }),
  );
}

async function fetchJson<T>(url: URL): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API thời tiết không khả dụng (${response.status}).`);
  }
  return (await response.json()) as T;
}

async function validateModelConfig(config: ModelConfig): Promise<ModelConfig> {
  const featureColumns = await fetchJson<string[]>(
    new URL(modelAssetUrl("airwise_feature_columns.json")),
  );
  const validScales = config.mean.every(Number.isFinite) &&
    config.scale.every((scale) => Number.isFinite(scale) && scale > 0);
  const matchingFeatures =
    featureColumns.length === config.feature_columns.length &&
    featureColumns.every(
      (column, index) => column === config.feature_columns[index],
    );

  if (
    config.feature_columns.length !== 29 ||
    config.mean.length !== config.feature_columns.length ||
    config.scale.length !== config.feature_columns.length ||
    !validScales ||
    !matchingFeatures
  ) {
    throw new Error(
      "Cấu hình scaler không khớp thứ tự 29 đặc trưng của mô hình.",
    );
  }
  return config;
}

async function loadAirSnapshot(
  coordinates: Coordinates,
): Promise<AirSnapshot> {
  const params = new URLSearchParams({
    latitude: coordinates.lat.toFixed(5),
    longitude: coordinates.lon.toFixed(5),
    timezone: "auto",
    past_days: "2",
    forecast_days: "1",
  });
  const weatherParams = new URLSearchParams(params);
  weatherParams.set(
    "hourly",
    "temperature_2m,dew_point_2m,pressure_msl,rain,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
  );
  weatherParams.set("wind_speed_unit", "ms");

  const airParams = new URLSearchParams(params);
  airParams.set(
    "hourly",
    "pm2_5,pm10,sulphur_dioxide,nitrogen_dioxide,carbon_monoxide,ozone",
  );

  const [weather, air] = await Promise.all([
    fetchJson<WeatherResponse>(
      new URL(`${weatherEndpoint}?${weatherParams.toString()}`),
    ),
    fetchJson<AirResponse>(new URL(`${airEndpoint}?${airParams.toString()}`)),
  ]);
  const config = await getModelConfig();
  const localNow = Date.now() + weather.utc_offset_seconds * 1000;

  let selectedIndex = -1;
  let candidateError: unknown;
  for (let index = weather.hourly.time.length - 1; index >= 24; index -= 1) {
    const time = weather.hourly.time[index];
    const [date, clock] = time.split("T");
    const [year, month, day] = date.split("-").map(Number);
    const [hour] = clock.split(":").map(Number);
    const timestamp = Date.UTC(year, month - 1, day, hour);
    if (timestamp <= localNow && localNow - timestamp < 3 * 60 * 60 * 1000) {
      try {
        buildInputVector(weather, air, index, config);
        selectedIndex = index;
        break;
      } catch (error) {
        candidateError = error;
        continue;
      }
    }
  }
  if (selectedIndex < 0) {
    const detail =
      candidateError instanceof Error ? ` ${candidateError.message}` : "";
    throw new Error(
      `Chưa có đủ dữ liệu thời tiết và PM2.5 theo giờ tại vị trí này.${detail} Hãy thử vị trí khác hoặc quay lại sau.`,
      { cause: candidateError },
    );
  }

  const input = buildInputVector(weather, air, selectedIndex, config);
  const runtime = await getRuntime();
  const inputTensor = new runtime.Tensor("float32", input, [1, input.length]);
  const predictions = await Promise.all(
    Object.entries(modelNames).map(async ([key, name]) => {
      const session = await getSession(name);
      const output = await session.run({
        [session.inputNames[0]]: inputTensor,
      });
      return [
        key,
        Number(output[session.outputNames[0]].data[0]),
      ] as const;
    }),
  );
  const predictionMap = Object.fromEntries(predictions) as Record<
    keyof typeof modelNames,
    number
  >;
  const pm25 = Math.max(0, predictionMap.point);
  const lower = Math.max(
    0,
    Math.min(predictionMap.lower, predictionMap.upper),
  );
  const upper = Math.max(predictionMap.lower, predictionMap.upper);
  const selectedAirIndex = air.hourly.time.indexOf(
    weather.hourly.time[selectedIndex],
  );
  const humidity = getValue(
    weather.hourly.relative_humidity_2m[selectedIndex],
    "độ ẩm",
  );

  return {
    coordinates,
    pm25,
    lower,
    upper,
    aqi: 0,
    temperature: getValue(
      weather.hourly.temperature_2m[selectedIndex],
      "nhiệt độ",
    ),
    windSpeed: getValue(
      weather.hourly.wind_speed_10m[selectedIndex],
      "tốc độ gió",
    ),
    humidity,
    rain: getValue(weather.hourly.rain[selectedIndex], "lượng mưa"),
    pressure: getValue(
      weather.hourly.pressure_msl[selectedIndex],
      "áp suất",
    ),
    measuredAt: weather.hourly.time[selectedIndex],
    history: Array.from({ length: 24 }, (_, index) => {
      const historyIndex = selectedAirIndex - 24 + index;
      return {
        time: air.hourly.time[historyIndex],
        pm25: getValue(air.hourly.pm2_5[historyIndex], "PM2.5 lịch sử"),
      };
    }),
  };
}

export function fetchAirSnapshot(
  coordinates: Coordinates,
): Promise<AirSnapshot> {
  const key = `${coordinates.lat.toFixed(5)},${coordinates.lon.toFixed(5)}`;
  const cached = snapshotCache.get(key);
  if (cached && cached.expiresAt > Date.now()) {
    return Promise.resolve(cached.snapshot);
  }

  const pendingRequest = snapshotRequests.get(key);
  if (pendingRequest) return pendingRequest;

  const request = loadAirSnapshot(coordinates)
    .then((snapshot) => {
      snapshotCache.set(key, {
        snapshot,
        expiresAt: Date.now() + snapshotCacheMs,
      });
      return snapshot;
    })
    .finally(() => {
      snapshotRequests.delete(key);
    });
  snapshotRequests.set(key, request);
  return request;
}

