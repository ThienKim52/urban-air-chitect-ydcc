export type AirQualityLevel = {
  label: string;
  color: string;
  description: string;
};

const breakpoints = [
  { concentrationLow: 0, concentrationHigh: 12, aqiLow: 0, aqiHigh: 50 },
  { concentrationLow: 12.1, concentrationHigh: 35.4, aqiLow: 51, aqiHigh: 100 },
  { concentrationLow: 35.5, concentrationHigh: 55.4, aqiLow: 101, aqiHigh: 150 },
  { concentrationLow: 55.5, concentrationHigh: 150.4, aqiLow: 151, aqiHigh: 200 },
  { concentrationLow: 150.5, concentrationHigh: 250.4, aqiLow: 201, aqiHigh: 300 },
  { concentrationLow: 250.5, concentrationHigh: 350.4, aqiLow: 301, aqiHigh: 400 },
  { concentrationLow: 350.5, concentrationHigh: 500.4, aqiLow: 401, aqiHigh: 500 },
];

export function calculateAqi(pm25: number): number {
  if (!Number.isFinite(pm25)) {
    throw new Error("Nồng độ PM2.5 phải là một số hợp lệ.");
  }
  const concentration =
    Math.floor(Math.max(0, Math.min(500.4, pm25)) * 10) / 10;
  const breakpoint = breakpoints.find(
    (range) =>
      concentration >= range.concentrationLow &&
      concentration <= range.concentrationHigh,
  )!;

  return Math.round(
    ((breakpoint.aqiHigh - breakpoint.aqiLow) /
      (breakpoint.concentrationHigh - breakpoint.concentrationLow)) *
      (concentration - breakpoint.concentrationLow) +
      breakpoint.aqiLow,
  );
}

export function getAirQualityLevel(aqi: number): AirQualityLevel {
  if (aqi <= 50) {
    return {
      label: "Tốt",
      color: "#2b9068",
      description: "Chất lượng không khí tốt, thích hợp cho các hoạt động ngoài trời.",
    };
  }
  if (aqi <= 100) {
    return {
      label: "Trung bình",
      color: "#d0a128",
      description: "Không khí ở mức chấp nhận được; người nhạy cảm nên hạn chế gắng sức.",
    };
  }
  if (aqi <= 150) {
    return {
      label: "Không tốt cho nhóm nhạy cảm",
      color: "#d27838",
      description: "Trẻ em, người cao tuổi và người có bệnh hô hấp nên giảm hoạt động ngoài trời.",
    };
  }
  if (aqi <= 200) {
    return {
      label: "Không tốt",
      color: "#d45252",
      description: "Hạn chế hoạt động ngoài trời kéo dài; nhóm nhạy cảm nên ở trong nhà.",
    };
  }
  if (aqi <= 300) {
    return {
      label: "Rất không tốt",
      color: "#9157a1",
      description: "Nên ở trong nhà và đóng cửa sổ; tránh hoạt động ngoài trời.",
    };
  }
  return {
    label: "Nguy hại",
    color: "#7c354b",
    description: "Hạn chế ra ngoài; nếu phải ra ngoài, cân nhắc dùng khẩu trang phù hợp.",
  };
}
