import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "../..");
const sourceDirectory = resolve(projectRoot, "models");
const targetDirectory = resolve(projectRoot, "webapp/public/models");
const assets = [
  "xgboost_pm25_point.onnx",
  "xgboost_pm25_q05.onnx",
  "xgboost_pm25_q95.onnx",
  "airwise_scaler.json",
  "airwise_feature_columns.json",
];

await mkdir(targetDirectory, { recursive: true });

for (const asset of assets) {
  const source = resolve(sourceDirectory, asset);
  try {
    await copyFile(source, resolve(targetDirectory, asset));
  } catch (error) {
    throw new Error(
      `Could not copy ${source}. Ensure the exported AirWise model assets exist.`,
      { cause: error },
    );
  }
}

console.log(`Prepared ${assets.length} AirWise model assets.`);
