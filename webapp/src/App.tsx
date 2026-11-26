import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Check,
  Clock3,
  CloudRain,
  Droplets,
  Gauge,
  LocateFixed,
  MapPin,
  Navigation,
  Route as RouteIcon,
  ShieldCheck,
  Sparkles,
  Wind,
} from "lucide-react";
import type { CSSProperties } from "react";
import AirMap from "./components/AirMap";
import TrendChart from "./components/TrendChart";
import { calculateAqi, getAirQualityLevel } from "./domain/aqi";
import { fetchAirSnapshot, type AirSnapshot } from "./domain/airQuality";
import { findRoutes } from "./domain/routing";
import type { Coordinates, RouteAlternative } from "./domain/types";

const initialLocation = { lat: 39.9042, lon: 116.4074 };
const initialOrigin = { lat: 39.9296, lon: 116.3956 };
const initialDestination = { lat: 39.8794, lon: 116.4551 };

function formatCoordinate(coordinate: Coordinates): string {
  return `${coordinate.lat.toFixed(3)}, ${coordinate.lon.toFixed(3)}`;
}

function formatTime(value: string): string {
  const [date, time] = value.split("T");
  return `${time.slice(0, 5)} · ${date}`;
}

function formatDuration(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} phút`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes > 0
    ? `${hours} giờ ${remainingMinutes} phút`
    : `${hours} giờ`;
}

function formatDistance(meters: number): string {
  return meters >= 1000
    ? `${(meters / 1000).toFixed(1)} km`
    : `${Math.round(meters)} m`;
}

function App() {
  const [location, setLocation] = useState<Coordinates>(initialLocation);
  const [snapshot, setSnapshot] = useState<AirSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [origin, setOrigin] = useState<Coordinates>(initialOrigin);
  const [destination, setDestination] =
    useState<Coordinates>(initialDestination);
  const [pickMode, setPickMode] = useState<
    "location" | "origin" | "destination" | null
  >(null);
  const [routes, setRoutes] = useState<RouteAlternative[]>([]);
  const [selectedRoute, setSelectedRoute] = useState(0);
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    setSnapshot(null);

    fetchAirSnapshot(location)
      .then((result) => {
        if (active) {
          setSnapshot({ ...result, aqi: calculateAqi(result.pm25) });
        }
      })
      .catch((cause: unknown) => {
        if (active) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Không thể tải dữ liệu chất lượng không khí.",
          );
          setSnapshot(null);
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [location]);

  const handlePickLocation = useCallback(
    (coordinates: Coordinates) => {
      if (pickMode === "origin") {
        setOrigin(coordinates);
        setRoutes([]);
        setRouteError("");
      } else if (pickMode === "destination") {
        setDestination(coordinates);
        setRoutes([]);
        setRouteError("");
      } else if (pickMode === "location") {
        setLocation(coordinates);
      }
      setPickMode(null);
    },
    [pickMode],
  );

  const handleSelectRoute = useCallback((index: number) => {
    setSelectedRoute(index);
  }, []);

  const airQuality = useMemo(
    () => (snapshot ? getAirQualityLevel(snapshot.aqi) : null),
    [snapshot],
  );

  async function handleFindRoutes() {
    setRouteLoading(true);
    setRouteError("");
    setRoutes([]);
    try {
      const foundRoutes = await findRoutes(origin, destination);
      foundRoutes.sort((first, second) => first.exposure - second.exposure);
      setRoutes(foundRoutes);
      setSelectedRoute(0);
    } catch (cause) {
      setRouteError(
        cause instanceof Error
          ? cause.message
          : "Không thể so sánh các tuyến đường.",
      );
    } finally {
      setRouteLoading(false);
    }
  }

  function handleLocateMe() {
    if (!navigator.geolocation) {
      setError("Trình duyệt này không hỗ trợ xác định vị trí.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        });
      },
      (cause) => {
        setError(
          cause.code === cause.PERMISSION_DENIED
            ? "Bạn chưa cấp quyền truy cập vị trí. Hãy chọn một điểm trên bản đồ."
            : "Không thể xác định vị trí hiện tại. Hãy chọn một điểm trên bản đồ.",
        );
      },
      { enableHighAccuracy: false, timeout: 10000 },
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="AirWise trang chủ">
          <span className="brand-mark">
            <Wind size={20} strokeWidth={2.2} />
          </span>
          <span>airwise<span className="brand-period">.</span></span>
        </a>
        <nav className="topnav" aria-label="Điều hướng chính">
          <a className="topnav-link topnav-link--active" href="#overview">Tổng quan</a>
          <a className="topnav-link" href="#map">Bản đồ</a>
          <a className="topnav-link" href="#routes">Lộ trình</a>
        </nav>
        <div className="privacy-badge">
          <ShieldCheck size={15} />
          <span>Mô hình chạy trên thiết bị</span>
        </div>
      </header>

      <main className="dashboard">
        <section className="welcome-row" id="overview">
          <div>
            <p className="eyebrow"><span className="live-dot" /> KHÔNG KHÍ · THEO THỜI GIAN THỰC</p>
            <h1>Hít thở dễ dàng hơn.</h1>
            <p className="welcome-copy">
              Chất lượng không khí và lộ trình thông minh, dành riêng cho bạn.
            </p>
          </div>
          <button className="location-button" onClick={handleLocateMe} type="button">
            <LocateFixed size={17} />
            Vị trí của tôi
          </button>
        </section>

        {error && (
          <div className="notice notice--error" role="alert">
            <span>{error}</span>
            <button type="button" onClick={() => setLocation({ ...location })}>
              Thử lại
            </button>
          </div>
        )}

        <section className="overview-grid" aria-label="Tổng quan chất lượng không khí">
          <article
            className={`air-card${airQuality ? "" : " air-card--loading"}`}
            style={
              airQuality
                ? ({ "--aq-color": airQuality.color } as CSSProperties)
                : undefined
            }
          >
            <div className="air-card-top">
              <div className="air-icon"><Activity size={19} /></div>
              <span className="air-card-label">CHỈ SỐ CHẤT LƯỢNG KHÔNG KHÍ</span>
              {loading ? (
                <span className="status-pill status-pill--loading">Đang cập nhật</span>
              ) : airQuality ? (
                <span className="status-pill">
                  <span className="status-dot" />
                  {airQuality.label}
                </span>
              ) : null}
            </div>
            <div className="aqi-reading">
              <span className="aqi-number">
                {loading ? <span className="skeleton skeleton-number" /> : snapshot?.aqi ?? "—"}
              </span>
              <span className="aqi-unit">AQI</span>
            </div>
            <p className="aqi-description">
              {loading
                ? "Đang lấy dữ liệu thời tiết và khởi tạo mô hình."
                : airQuality?.description ?? "Chưa có dữ liệu tại vị trí này."}
            </p>
            <div className="air-card-footer">
              <span><MapPin size={14} /> {formatCoordinate(location)}</span>
              {snapshot && <span><Clock3 size={14} /> {formatTime(snapshot.measuredAt)}</span>}
            </div>
          </article>

          <article className="metric-card">
            <div className="metric-heading"><span className="metric-icon metric-icon--amber"><Wind size={17} /></span><span>Nhiệt độ & gió</span></div>
            <div className="metric-value">
              {loading ? "—" : snapshot ? `${snapshot.temperature.toFixed(1)}°` : "—"}
              <span>{snapshot ? `${snapshot.windSpeed.toFixed(1)} m/s` : "·"}</span>
            </div>
            <p className="metric-caption">Nhiệt độ <span>·</span> Tốc độ gió</p>
            <div className="metric-track"><span className="metric-track-fill metric-track-fill--amber" /></div>
          </article>

          <article className="metric-card">
            <div className="metric-heading"><span className="metric-icon metric-icon--blue"><Droplets size={17} /></span><span>Độ ẩm & mưa</span></div>
            <div className="metric-value">
              {loading ? "—" : snapshot ? `${Math.round(snapshot.humidity)}%` : "—"}
              <span>{snapshot ? `${snapshot.rain.toFixed(1)} mm` : "·"}</span>
            </div>
            <p className="metric-caption">Độ ẩm tương đối <span>·</span> Lượng mưa</p>
            <div className="metric-track"><span className="metric-track-fill metric-track-fill--blue" /></div>
          </article>

          <article className="metric-card metric-card--model">
            <div className="metric-heading"><span className="metric-icon metric-icon--green"><Sparkles size={17} /></span><span>Dự báo AI</span></div>
            {loading ? (
              <div className="model-loading"><span className="skeleton skeleton-line" /><span className="skeleton skeleton-line skeleton-line--short" /></div>
            ) : snapshot ? (
              <>
                <div className="metric-value metric-value--range">
                  <span>{snapshot.lower.toFixed(0)}–{snapshot.upper.toFixed(0)}</span>
                  <span>µg/m³</span>
                </div>
                <p className="metric-caption">Khoảng dự báo PM2.5 · phân vị 5–95%</p>
              </>
            ) : (
              <p className="metric-caption metric-caption--message">Đang chờ dữ liệu khả dụng</p>
            )}
            <div className="model-tag"><span className="model-tag-dot" /> XGBoost · ONNX Runtime Web</div>
          </article>
        </section>

        <section className="workspace-grid">
          <article className="panel map-panel" id="map">
            <div className="panel-heading">
              <div>
                <div className="panel-title-row"><span className="panel-title-icon"><MapPin size={17} /></span><h2>Bản đồ chất lượng không khí</h2></div>
                  <p className="panel-subtitle">Chọn vị trí dự báo hoặc đặt điểm đi, điểm đến trên bản đồ.</p>
              </div>
              <span className="map-source"><span className="live-dot" /> CAMS · Open-Meteo</span>
            </div>
            <div className={`map-frame${pickMode ? " map-frame--picking" : ""}`}>
              <AirMap
                location={location}
                qualityColor={airQuality?.color ?? "#73917d"}
                aqi={snapshot?.aqi ?? null}
                origin={origin}
                destination={destination}
                routes={routes}
                selectedRoute={selectedRoute}
                pickMode={pickMode}
                onPickLocation={handlePickLocation}
                onSelectRoute={handleSelectRoute}
              />
              {pickMode && (
                <div className="map-pick-banner">
                  <MapPin size={15} />
                  Chọn {pickMode === "location" ? "vị trí dự báo" : pickMode === "origin" ? "điểm đi" : "điểm đến"} trên bản đồ
                  <button type="button" onClick={() => setPickMode(null)}>Hủy</button>
                </div>
              )}
              {loading && (
                <div className="map-loading"><span className="spinner" /> Đang tải dự báo...</div>
              )}
            </div>
            <div className="map-legend">
              <span><i className="legend-dot legend-dot--location" style={{ backgroundColor: airQuality?.color ?? "#73917d" }} /> Vị trí dự báo</span>
              <span><i className="legend-line" /> Tuyến ít phơi nhiễm nhất</span>
              <button type="button" onClick={() => setPickMode("location")}><Sparkles size={14} /> Dự báo tại đây</button>
              <button type="button" onClick={handleLocateMe}><LocateFixed size={14} /> Đến vị trí của tôi</button>
            </div>
          </article>

          <aside className="side-stack">
            <article className="panel trend-panel" id="forecast">
              <div className="panel-heading">
                <div>
                  <div className="panel-title-row"><span className="panel-title-icon panel-title-icon--gold"><Activity size={16} /></span><h2>PM2.5 trong 24 giờ</h2></div>
                  <p className="panel-subtitle">Quan trắc quá khứ và dự báo AI hiện tại</p>
                </div>
                <span className="unit-tag">µg/m³</span>
              </div>
              {snapshot ? (
                <>
                  <TrendChart snapshot={snapshot} />
                  <div className="chart-legend">
                    <span><i className="chart-legend-line" /> PM2.5 · Open-Meteo</span>
                    <span><i className="chart-legend-dot" /> Dự báo AI</span>
                  </div>
                </>
              ) : (
                <div className="empty-chart">
                  {loading ? <span className="spinner" /> : <Activity size={22} />}
                  <span>{loading ? "Đang tải dữ liệu…" : "Chưa có dữ liệu biểu đồ"}</span>
                </div>
              )}
            </article>

            <article className="advice-card">
              <div className="advice-heading"><span className="advice-icon"><ShieldCheck size={17} /></span><span>Gợi ý cho sức khỏe</span></div>
              <p>{airQuality ? airQuality.description : "Chọn một vị trí có dữ liệu để nhận khuyến nghị phù hợp."}</p>
              <span className="advice-note">Tham khảo chỉ số, không thay thế tư vấn y tế.</span>
            </article>
          </aside>
        </section>

        <section className="panel route-panel" id="routes">
          <div className="route-panel-heading">
            <div>
              <div className="panel-title-row"><span className="panel-title-icon panel-title-icon--route"><RouteIcon size={17} /></span><h2>Tìm lộ trình ít bụi hơn</h2></div>
              <p className="panel-subtitle">So sánh lượng PM2.5 dự báo dọc theo các tuyến đường thay thế.</p>
            </div>
            <span className="route-provider"><Navigation size={14} /> OSRM · OpenStreetMap</span>
          </div>
          <div className="route-controls">
            <div className="route-point">
              <span className="route-point-dot route-point-dot--start" />
              <div className="route-point-info"><span>ĐIỂM ĐI</span><strong>{formatCoordinate(origin)}</strong></div>
              <button type="button" onClick={() => setPickMode("origin")} className={pickMode === "origin" ? "point-button point-button--active" : "point-button"}>
                <MapPin size={14} /> Chọn trên bản đồ
              </button>
            </div>
            <span className="route-connector"><ArrowRight size={16} /></span>
            <div className="route-point">
              <span className="route-point-dot route-point-dot--end" />
              <div className="route-point-info"><span>ĐIỂM ĐẾN</span><strong>{formatCoordinate(destination)}</strong></div>
              <button type="button" onClick={() => setPickMode("destination")} className={pickMode === "destination" ? "point-button point-button--active" : "point-button"}>
                <MapPin size={14} /> Chọn trên bản đồ
              </button>
            </div>
            <button
              className="route-submit"
              type="button"
              onClick={handleFindRoutes}
              disabled={routeLoading}
            >
              {routeLoading ? <span className="spinner spinner--light" /> : <RouteIcon size={16} />}
              {routeLoading ? "Đang so sánh" : "So sánh tuyến"}
            </button>
          </div>
          {routeError && <p className="route-error" role="alert">{routeError}</p>}
          {routes.length > 0 && (
            <div className="route-results" aria-live="polite">
              {routes.map((route, index) => (
                <button
                  className={`route-result${selectedRoute === index ? " route-result--selected" : ""}`}
                  key={`${route.distance}-${route.duration}`}
                  onClick={() => setSelectedRoute(index)}
                  type="button"
                >
                  <span className="route-result-rank">{index === 0 ? <Check size={15} /> : index + 1}</span>
                  <span className="route-result-description">
                    <strong>{index === 0 ? "Ít phơi nhiễm nhất" : `Tuyến thay thế ${index}`}</strong>
                    <span>{formatDistance(route.distance)} <i /> {formatDuration(route.duration)}</span>
                  </span>
                  <span className="route-result-pm">
                    <strong>{route.averagePm25.toFixed(1)}</strong>
                    <span>µg/m³ PM2.5 TB</span>
                  </span>
                  <span className="route-result-exposure">
                    <strong>{route.exposure.toFixed(1)}</strong>
                    <span>µg/m³ × giờ</span>
                  </span>
                  {index === 0 ? <ArrowDownRight className="route-result-arrow" size={18} /> : <ArrowUpRight className="route-result-arrow" size={18} />}
                </button>
              ))}
              <p className="route-disclaimer"><CloudRain size={13} /> Ước lượng tham khảo từ PM2.5 của Open-Meteo tại 3 điểm trên mỗi tuyến; điều kiện thực tế có thể khác.</p>
            </div>
          )}
          {!routes.length && !routeLoading && !routeError && (
            <div className="route-empty"><span className="route-empty-icon"><MapPin size={17} /></span><span>Chọn hai điểm trên bản đồ để bắt đầu so sánh.</span><button type="button" onClick={() => setPickMode("origin")}>Chọn điểm đi <ArrowRight size={14} /></button></div>
          )}
        </section>

        <footer className="footer">
          <span>AirWise <i>·</i> Dữ liệu ước lượng, chỉ dùng cho mục đích tham khảo.</span>
          <span><Gauge size={13} /> Mô hình được huấn luyện trên dữ liệu không khí Bắc Kinh.</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
