import { networkInterfaces } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const envPath = resolve(scriptDir, "..", ".env.local");

function isPrivateLanIp(ip) {
  if (ip.startsWith("10.")) {
    return true;
  }

  if (ip.startsWith("192.168.")) {
    return true;
  }

  const match = ip.match(/^172\.(\d+)\./);
  return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
}

function getCurrentIpv4() {
  const candidates = Object.entries(networkInterfaces())
    .flatMap(([name, addresses]) =>
      (addresses || [])
        .filter((address) => address.family === "IPv4" && !address.internal)
        .map((address) => ({
          name,
          ip: address.address,
        })),
    )
    .sort((left, right) => getAdapterScore(right) - getAdapterScore(left));

  return candidates[0]?.ip || "127.0.0.1";
}

function getAdapterScore(candidate) {
  const name = candidate.name.toLowerCase();
  let score = 0;

  if (isPrivateLanIp(candidate.ip)) {
    score += 50;
  }

  if (/(wi-?fi|wireless|wlan|ethernet)/i.test(name)) {
    score += 30;
  }

  if (/(vmware|virtual|vbox|hyper-v|docker|wsl|loopback|vethernet)/i.test(name)) {
    score -= 60;
  }

  return score;
}

function writeEnvValue(apiBaseUrl) {
  const nextLine = `VITE_API_BASE_URL=${apiBaseUrl}`;
  const currentContent = existsSync(envPath) ? readFileSync(envPath, "utf8") : "";
  const lines = currentContent
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "" && !line.startsWith("VITE_API_BASE_URL="));

  lines.push(nextLine);
  writeFileSync(envPath, `${lines.join("\n")}\n`);
}

const ipv4 = getCurrentIpv4();
const apiBaseUrl = `http://${ipv4}:8000`;

writeEnvValue(apiBaseUrl);
console.log(`VITE_API_BASE_URL=${apiBaseUrl}`);
